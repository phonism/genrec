"""
Amazon LCRec Dataset for LCRec model training.
Provides semantic IDs as codebook tokens and text descriptions for LLM fine-tuning.

LCRec uses 6 training tasks:
- seqrec: Given history, predict next item's codebook tokens
- item2index: Given item description, predict codebook tokens
- index2item: Given codebook tokens, predict item description
- fusionseqrec: Given history, predict next item's tokens AND title
- itemsearch: Given query + history, find matching item
- preferenceobtain: Given history, infer user preferences
"""
import os
import gin
import torch
import random

from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Set

from genrec.data.amazon import (
    AmazonItemDataset,
    DATASET_CONFIGS,
    parse_gzip_json,
)


# SFT prompt format (matching official LC-Rec implementation)
SFT_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

# History item separator (matching official LC-Rec)
HISTORY_SEP = ", "  # Official uses ", " separator
ADD_PREFIX = True  # Add "1. ", "2. ", etc. prefix to history items


# Prompt templates for each task (sampled randomly during training)
# Based on official LC-Rec implementation with expanded variations
PROMPT_TEMPLATES = {
    # Sequential Recommendation - 17 templates
    "seqrec": [
        "User interaction history: {history}\nPredict the next item:",
        "Given the user's past interactions: {history}\nWhat item will they interact with next?",
        "The user has interacted with: {history}\nRecommend the next item:",
        "Based on history: {history}\nNext item prediction:",
        "A user has the following purchase history: {history}\nCan you predict the next possible item?",
        "Here is the interaction history of a user: {history}\nPlease predict what item they would like next.",
        "Given a user's historical interactions: {history}\nWhat will be their next interaction?",
        "The user previously bought: {history}\nPredict the next purchase:",
        "User's sequential behavior: {history}\nNext item:",
        "Past items: {history}\nFuture item prediction:",
        "Interaction sequence: {history}\nRecommend next:",
        "Purchase history: {history}\nNext likely item:",
        "User browsing history: {history}\nPredict next item of interest:",
        "Historical purchases: {history}\nSuggest next item:",
        "Given user history: {history}\nWhat should we recommend next?",
        "User activity: {history}\nNext item they might like:",
        "Please predict the next item that the user most desires based on: {history}\nNext item:",
    ],
    # Item to Index - title-based templates
    "item2index_title": [
        "Item title: {title}\nItem index:",
        "Given the item titled \"{title}\", what is its index?",
        "Find the index for item: {title}\nIndex:",
        "What is the index of the item with title: {title}?",
        "Item named \"{title}\" has index:",
        "Look up index for: {title}\nResult:",
    ],
    # Item to Index - description-based templates
    "item2index_desc": [
        "Item description: {description}\nItem index:",
        "Given item with description: {description}\nWhat is its index?",
        "Find index for item described as: {description}\nIndex:",
        "An item described as \"{description}\" has index:",
        "What index corresponds to: {description}?",
        "Item matching description \"{description}\" has index:",
    ],
    # Item to Index - combined templates
    "item2index_combined": [
        "Item: {title} - {description}\nItem index:",
        "Given item \"{title}\" with description \"{description}\", find its index:",
        "What is the index of: {title} ({description})?",
        "Find index for \"{title}\": {description}\nIndex:",
        "Item titled \"{title}\" described as \"{description}\" has index:",
        "Look up: {title} - {description}\nIndex:",
        "Match item \"{title}\" with features: {description}\nIndex:",
    ],
    # Index to Item - title templates
    "index2item_title": [
        "Item index: {index}\nItem title:",
        "Given index {index}, what is the item's title?",
        "Find the title for item with index: {index}",
        "What item has index {index}? Title:",
        "Index {index} corresponds to item titled:",
        "Retrieve title for index: {index}\nTitle:",
    ],
    # Index to Item - description templates
    "index2item_desc": [
        "Item index: {index}\nItem description:",
        "Given index {index}, describe the item:",
        "What is the description of item with index: {index}?",
        "Index {index} represents an item described as:",
        "Describe the item at index: {index}\nDescription:",
        "Item description for index {index}:",
    ],
    # Index to Item - combined templates
    "index2item_combined": [
        "Item index: {index}\nItem title and description:",
        "Given index {index}, what is the item's title and description?",
        "Describe item at index: {index}\nTitle and description:",
        "What item has index {index}? Provide title and description:",
        "Index {index} corresponds to:",
    ],
    # Fusion Sequential Recommendation - 12 templates
    "fusionseqrec": [
        "User interaction history: {history}\nPredict the next item index and title:",
        "Given history: {history}\nRecommend next item with its name:",
        "Based on: {history}\nNext item (index and title):",
        "History: {history}\nPredict next item's index and name:",
        "User has interacted with: {history}\nNext item index and description:",
        "Past interactions: {history}\nPredict next item's identifier and title:",
        "Sequential history: {history}\nWhat's next? Provide index and title:",
        "Given user behavior: {history}\nNext item (index, title):",
        "Purchase sequence: {history}\nRecommend with item details:",
        "User's history: {history}\nNext recommendation (index and name):",
        "Based on interactions: {history}\nPredict next item with full info:",
        "History of items: {history}\nNext item prediction with title:",
    ],
    # Item Search - 11 templates
    "itemsearch": [
        "User wants: {query}\nHistory: {history}\nFind matching item:",
        "Search query: {query}\nUser's past items: {history}\nBest match:",
        "Query: {query}\nBased on history: {history}\nRecommended item:",
        "User is looking for: {query}\nPrevious interactions: {history}\nSuggestion:",
        "Search: {query}\nUser preferences from: {history}\nResult:",
        "Find item matching: {query}\nUser context: {history}\nItem:",
        "User searches for: {query}\nGiven their history: {history}\nBest item:",
        "Query: {query}\nPersonalized by history: {history}\nMatch:",
        "Looking for: {query}\nBased on past behavior: {history}\nRecommendation:",
        "Search intent: {query}\nUser profile from: {history}\nSuggested item:",
        "User query: {query}\nInteraction history: {history}\nBest matching item:",
    ],
    # Preference Obtainment - 12 templates
    "preferenceobtain": [
        "User interaction history: {history}\nInfer user preferences:",
        "Based on items: {history}\nWhat are the user's preferences?",
        "Given history: {history}\nDescribe user's taste:",
        "Items interacted: {history}\nUser preference analysis:",
        "From purchase history: {history}\nSummarize user interests:",
        "User has bought: {history}\nWhat do they prefer?",
        "Analyzing history: {history}\nUser preferences are:",
        "Given interactions: {history}\nInfer shopping preferences:",
        "User behavior: {history}\nPreference summary:",
        "Historical items: {history}\nUser's interests include:",
        "Based on: {history}\nThe user is interested in:",
        "User's past items: {history}\nPreference profile:",
    ],
}


@gin.configurable
class AmazonLCRecDataset(Dataset):
    """
    Amazon Dataset for LCRec training.

    Provides training samples for 6 tasks:
    - seqrec: sequence recommendation (history -> next item index)
    - item2index: item description -> index
    - index2item: index -> item description
    - fusionseqrec: history -> next item index + title
    - itemsearch: query + history -> item index
    - preferenceobtain: history -> preference description
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        max_text_len: int = 128,
        pretrained_rqvae_path: str = "./out/lcrec/amazon/{split}/rqvae/checkpoint.pt",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        # RQVAE config - should match pretrained model (official LC-Rec settings)
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 64,  # Official: 64
        rqvae_hidden_dims: List[int] = [512, 256, 128],  # Official config
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 5,  # Official: 5 codebooks
        # Task configuration
        enabled_tasks: List[str] = None,
        task_sample_weights: Dict[str, float] = None,
    ) -> None:
        from genrec.models.rqvae import RqVae

        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.n_codebooks = rqvae_n_layers
        self.codebook_size = rqvae_codebook_size

        # Default: enable all tasks
        self.enabled_tasks: Set[str] = set(enabled_tasks or [
            "seqrec", "item2index", "index2item",
            "fusionseqrec", "itemsearch", "preferenceobtain"
        ])

        # Task sampling weights (for training data generation)
        self.task_sample_weights = task_sample_weights or {
            "seqrec": 1.0,
            "item2index": 0.5,
            "index2item": 0.5,
            "fusionseqrec": 0.5,
            "itemsearch": 0.3,
            "preferenceobtain": 0.3,
        }

        # Replace {split} placeholder in rqvae path
        pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Load pretrained RQVAE and generate semantic IDs
        rqvae = RqVae(
            input_dim=rqvae_input_dim,
            embed_dim=rqvae_embed_dim,
            hidden_dims=rqvae_hidden_dims,
            codebook_size=rqvae_codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=False,
            codebook_sim_vq=False,
            n_layers=rqvae_n_layers,
            n_cat_features=0,
            commitment_weight=0.25,
        )
        rqvae.load_pretrained(pretrained_rqvae_path)
        rqvae.eval()

        with torch.no_grad():
            self.sem_ids_list = rqvae.get_semantic_ids(self.item_embeddings).sem_ids.tolist()

        # Load item metadata for text
        self._load_item_metadata()

        # Load user sequences and generate samples
        self._load_sequences()
        self._generate_samples()

    def _load_item_metadata(self) -> None:
        """Load item metadata for text generation."""
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.root, "raw", self.split, config["meta"])
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        # Build item mapping from reviews
        item_id_mapping: Dict[str, int] = {}
        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping)

        # Load metadata
        self.item_texts: Dict[int, str] = {}
        self.item_titles: Dict[int, str] = {}
        self.item_categories: Dict[int, str] = {}
        for meta in parse_gzip_json(meta_path):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                title = meta.get('title', '')
                brand = meta.get('brand', '')
                category = ', '.join(meta.get('categories', [[]])[-1][:3]) if meta.get('categories') else ''

                # Full text description
                text = f"{title}"
                if brand:
                    text += f" by {brand}"
                if category:
                    text += f" ({category})"
                text = text.strip() or f"item_{item_id}"

                self.item_texts[item_id] = text
                self.item_titles[item_id] = title or f"item_{item_id}"
                self.item_categories[item_id] = category

        # Fill missing items
        for i in range(len(item_id_mapping)):
            if i not in self.item_texts:
                self.item_texts[i] = f"item_{i}"
                self.item_titles[i] = f"item_{i}"
                self.item_categories[i] = ""

        self.num_items = len(item_id_mapping)

    def _load_sequences(self) -> None:
        """Load user interaction sequences from reviews."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping)

                item_id = item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort by timestamp and filter short sequences
        self.sequences = []
        self.user_ids = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)
                self.user_ids.append(uid)

        print(f"Loaded {len(self.sequences)} user sequences for LCRec")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples based on enabled tasks."""
        self.samples = []

        if self.train_test_split == "train":
            self._generate_train_samples()
        else:
            self._generate_eval_samples()

        print(f"Generated {len(self.samples)} LCRec samples for {self.train_test_split}")
        # Print task distribution
        task_counts = {}
        for s in self.samples:
            task_counts[s['task']] = task_counts.get(s['task'], 0) + 1
        print(f"Task distribution: {task_counts}")

    def _generate_train_samples(self) -> None:
        """Generate training samples for all enabled tasks."""
        # Sequence-based tasks
        for user_idx, full_seq in enumerate(self.sequences):
            seq = full_seq[:-2]  # Leave last 2 for valid/test
            if len(seq) < 2:
                continue

            # SeqRec samples (sliding window)
            if "seqrec" in self.enabled_tasks:
                for i in range(1, len(seq)):
                    history = seq[max(0, i - self._max_seq_len):i]
                    target = seq[i]
                    self.samples.append({
                        'task': 'seqrec',
                        'history': history,
                        'target': target,
                    })

            # FusionSeqRec samples
            if "fusionseqrec" in self.enabled_tasks:
                for i in range(1, len(seq)):
                    if random.random() < self.task_sample_weights.get("fusionseqrec", 0.5):
                        history = seq[max(0, i - self._max_seq_len):i]
                        target = seq[i]
                        self.samples.append({
                            'task': 'fusionseqrec',
                            'history': history,
                            'target': target,
                        })

            # ItemSearch samples (simulate search query from target item)
            if "itemsearch" in self.enabled_tasks:
                for i in range(1, len(seq)):
                    if random.random() < self.task_sample_weights.get("itemsearch", 0.3):
                        history = seq[max(0, i - self._max_seq_len):i]
                        target = seq[i]
                        self.samples.append({
                            'task': 'itemsearch',
                            'history': history,
                            'target': target,
                        })

            # PreferenceObtain samples
            if "preferenceobtain" in self.enabled_tasks:
                if random.random() < self.task_sample_weights.get("preferenceobtain", 0.3):
                    history = seq[-self._max_seq_len:]
                    self.samples.append({
                        'task': 'preferenceobtain',
                        'history': history,
                    })

        # Item-based tasks (all items) - with subtypes like official LC-Rec
        if "item2index" in self.enabled_tasks:
            for item_id in range(self.num_items):
                if item_id < len(self.sem_ids_list):
                    # Add all three subtypes: title, description, combined
                    for subtype in ['title', 'desc', 'combined']:
                        self.samples.append({
                            'task': 'item2index',
                            'item_id': item_id,
                            'subtype': subtype,
                        })

        if "index2item" in self.enabled_tasks:
            for item_id in range(self.num_items):
                if item_id < len(self.sem_ids_list):
                    # Add all three subtypes: title, description, combined
                    for subtype in ['title', 'desc', 'combined']:
                        self.samples.append({
                            'task': 'index2item',
                            'item_id': item_id,
                            'subtype': subtype,
                        })

    def _generate_eval_samples(self) -> None:
        """Generate evaluation samples (seqrec only for fair comparison)."""
        if self.train_test_split == "valid":
            for user_idx, full_seq in enumerate(self.sequences):
                seq = full_seq[:-1]
                if len(seq) >= 2:
                    history = seq[max(0, len(seq) - 1 - self._max_seq_len):-1]
                    target = seq[-1]
                    self.samples.append({
                        'task': 'seqrec',
                        'history': history,
                        'target': target,
                    })
        else:  # test
            for user_idx, full_seq in enumerate(self.sequences):
                if len(full_seq) >= 2:
                    history = full_seq[max(0, len(full_seq) - 1 - self._max_seq_len):-1]
                    target = full_seq[-1]
                    self.samples.append({
                        'task': 'seqrec',
                        'history': history,
                        'target': target,
                    })

    def _sem_ids_to_tokens(self, sem_ids: List[int]) -> str:
        """Convert semantic IDs to codebook token string."""
        tokens = []
        for c, code in enumerate(sem_ids):
            tokens.append(f"<C{c}_{code}>")
        return "".join(tokens)

    def _history_to_tokens(self, history: List[int]) -> str:
        """Convert history item IDs to token string with numbered prefixes (official LC-Rec format)."""
        tokens = []
        for idx, item_id in enumerate(history):
            if item_id < len(self.sem_ids_list):
                item_token = self._sem_ids_to_tokens(self.sem_ids_list[item_id])
            else:
                item_token = "<UNK>"
            if ADD_PREFIX:
                tokens.append(f"{idx + 1}. {item_token}")
            else:
                tokens.append(item_token)
        return HISTORY_SEP.join(tokens)

    def _get_random_prompt(self, task: str) -> str:
        """Get a random prompt template for the task."""
        templates = PROMPT_TEMPLATES.get(task, PROMPT_TEMPLATES["seqrec"])
        return random.choice(templates)

    def _format_seqrec(self, history: List[int], target: int) -> Dict[str, str]:
        """Format sequence recommendation task using SFT format."""
        history_str = self._history_to_tokens(history)
        target_tokens = self._sem_ids_to_tokens(
            self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks
        )

        prompt_template = self._get_random_prompt("seqrec")
        instruction = prompt_template.format(history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        response = target_tokens

        return {"prompt": prompt, "response": response}

    def _format_item2index(self, item_id: int, subtype: str = 'title') -> Dict[str, str]:
        """Format item to index task with subtypes: title, desc, combined."""
        item_title = self.item_titles.get(item_id, f"item_{item_id}")
        item_text = self.item_texts.get(item_id, f"item_{item_id}")
        # Description is the full text minus the title part
        description = item_text.replace(item_title, "").strip(" -()") or item_title
        sem_ids = self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks

        template_key = f"item2index_{subtype}"
        prompt_template = self._get_random_prompt(template_key)

        if subtype == 'title':
            instruction = prompt_template.format(title=item_title)
        elif subtype == 'desc':
            instruction = prompt_template.format(description=description)
        else:  # combined
            instruction = prompt_template.format(title=item_title, description=description)

        prompt = SFT_PROMPT.format(instruction=instruction)
        response = self._sem_ids_to_tokens(sem_ids)
        return {"prompt": prompt, "response": response}

    def _format_index2item(self, item_id: int, subtype: str = 'title') -> Dict[str, str]:
        """Format index to item task with subtypes: title, desc, combined."""
        item_title = self.item_titles.get(item_id, f"item_{item_id}")
        item_text = self.item_texts.get(item_id, f"item_{item_id}")
        description = item_text.replace(item_title, "").strip(" -()") or item_title
        sem_ids = self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks
        index_str = self._sem_ids_to_tokens(sem_ids)

        template_key = f"index2item_{subtype}"
        prompt_template = self._get_random_prompt(template_key)
        instruction = prompt_template.format(index=index_str)
        prompt = SFT_PROMPT.format(instruction=instruction)

        if subtype == 'title':
            response = item_title
        elif subtype == 'desc':
            response = description
        else:  # combined
            response = f"{item_title}\n\n{description}"

        return {"prompt": prompt, "response": response}

    def _format_fusionseqrec(self, history: List[int], target: int) -> Dict[str, str]:
        """Format fusion sequence recommendation task (predict title/description)."""
        history_str = self._history_to_tokens(history)
        target_title = self.item_titles.get(target, f"item_{target}")
        target_text = self.item_texts.get(target, f"item_{target}")
        description = target_text.replace(target_title, "").strip(" -()") or target_title

        prompt_template = self._get_random_prompt("fusionseqrec")
        instruction = prompt_template.format(history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        # Response can be title or description depending on the prompt
        response = target_title

        return {"prompt": prompt, "response": response}

    def _format_itemsearch(self, history: List[int], target: int) -> Dict[str, str]:
        """Format item search task."""
        history_str = self._history_to_tokens(history)
        target_tokens = self._sem_ids_to_tokens(
            self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks
        )

        # Generate query from target item's title/category
        target_title = self.item_titles.get(target, "")
        target_category = self.item_categories.get(target, "")

        # Simulate a search query (use partial title or category)
        if target_category and random.random() < 0.5:
            query = target_category
        elif target_title:
            words = target_title.split()
            if len(words) > 2:
                query = " ".join(random.sample(words, min(3, len(words))))
            else:
                query = target_title
        else:
            query = "similar item"

        prompt_template = self._get_random_prompt("itemsearch")
        instruction = prompt_template.format(query=query, history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        response = target_tokens

        return {"prompt": prompt, "response": response}

    def _format_preferenceobtain(self, history: List[int]) -> Dict[str, str]:
        """Format preference obtainment task."""
        history_str = self._history_to_tokens(history)

        # Generate preference description from history items' categories
        categories = set()
        for item_id in history:
            cat = self.item_categories.get(item_id, "")
            if cat:
                categories.add(cat.split(",")[0].strip())

        if categories:
            preference = f"The user is interested in: {', '.join(list(categories)[:5])}"
        else:
            preference = "The user has diverse interests based on their interaction history."

        prompt_template = self._get_random_prompt("preferenceobtain")
        instruction = prompt_template.format(history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        response = preference

        return {"prompt": prompt, "response": response}

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        task = sample['task']

        if task == 'seqrec':
            formatted = self._format_seqrec(sample['history'], sample['target'])
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': sample['target'],
                'target_sem_ids': self.sem_ids_list[sample['target']] if sample['target'] < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'item2index':
            subtype = sample.get('subtype', 'title')
            formatted = self._format_item2index(sample['item_id'], subtype=subtype)
            item_id = sample['item_id']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': item_id,
                'target_sem_ids': self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'index2item':
            subtype = sample.get('subtype', 'title')
            formatted = self._format_index2item(sample['item_id'], subtype=subtype)
            item_id = sample['item_id']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': item_id,
                'target_sem_ids': self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'fusionseqrec':
            formatted = self._format_fusionseqrec(sample['history'], sample['target'])
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': sample['target'],
                'target_sem_ids': self.sem_ids_list[sample['target']] if sample['target'] < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'itemsearch':
            formatted = self._format_itemsearch(sample['history'], sample['target'])
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': sample['target'],
                'target_sem_ids': self.sem_ids_list[sample['target']] if sample['target'] < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'preferenceobtain':
            formatted = self._format_preferenceobtain(sample['history'])
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
            }
        else:
            raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Test dataset creation
    dataset = AmazonLCRecDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
        pretrained_rqvae_path="./out/lcrec/amazon/beauty/rqvae/checkpoint.pt",
        enabled_tasks=["seqrec", "item2index", "index2item"],  # Test with subset
    )
    print(f"Dataset size: {len(dataset)}")
    for i in range(min(5, len(dataset))):
        print(f"Sample {i}: {dataset[i]}")
