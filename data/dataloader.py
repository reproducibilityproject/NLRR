import torch
import random
import pickle
import logging
import numpy as np
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("------------------------------")
logger.info("Loading ACM Badges data......")

# set global seed
torch.manual_seed(2024)
np.random.seed(2024)

# helper function to set seed
def seed_worker(worker_id):
    np.random.seed(2024)
    random.seed(2024)

# base class to modify the dataframes into torch dataset
class ACMPapersDataset(Dataset):
    def __init__(self,
                 features, scaled_features,
                 auth_label_targets, ext_agent_label_targets,
                 joint_label_targets,
                 split_ratio=0.7):
        """
        Helper class to transform a given pair of features, and targets
        for a given torch tensor into a torch Dataset

        Parameters
        ----------
        arg1 | features: torch.tensor
            The features collected for a paper
        arg2 | scaled_features: torch.tensor
            The sclaed feature representations for a given paper
        arg3 | auth_label_targets: torch.tensor
            The author centric spectrum labels for a given paper in the dataset
        arg4 | ext_agent_label_targets: torch.tensor
            The external agent centric spectrum labels for a given paper in the dataset
        arg5 | joint_label_targets: torch.tensor
            The joint label for a given paper in the dataset
        arg6 | split_ratio: float
            The Split ratio necessary for training, and testing the model
        ----------
        Returns
        -------
        Torch Dataset
            torch.utils.data.Dataset

        """
        # input features of papers
        self.features = features

        # scaled feature inputs of papers
        self.scaled_features = scaled_features

        # target variable for author centric labels
        self.auth_label_targets = auth_label_targets

        # target variable for external agent centric labels
        self.ext_agent_label_targets = ext_agent_label_targets

        # target avariable for joint labels
        self.joint_label_targets = joint_label_targets

        # split ration to reserve sampels for training
        self.split_ratio = split_ratio

        # set training and test size
        train_size = int(len(self.features) * self.split_ratio)
        test_size = len(self.features) - train_size

        # prepare torch datasets
        self.train_ds, self.test_ds = random_split(self, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(2024))
    def __len__(self):
        return len(self.features)

    def transform_labels(self, labels, zero_index=True):
        # zero index labels type check for long
        if zero_index:
            labels = labels - 1

        # type check
        labels = labels.long()

        # return the one hot encoded targets
        return labels

    def train_dataloader(self, batch_size=32):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)

    def test_dataloader(self, batch_size=32):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    def __getitem__(self, idx):
        # return the item store with raw features, scaled features, and labels
        return {
            "X": self.features[idx], "scaled_X": self.scaled_features[idx],
            "y_auth": self.auth_label_targets[idx], "y_ext_agent": self.ext_agent_label_targets[idx],
            "y_joint": self.joint_label_targets[idx]
        }

# base class to modify text based dataframes into torch dataset
class ACMPaperTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, embeddings_file=None, max_len=512, split_ratio=0.7):
        """
        Helper class to transform a given dataframe, tokenizer, and targets
        for a given paper into a torch Dataset

        Parameters
        ----------
        arg1 | df: pandas.data.DataFrame
            The dataframe with full text store for a paper
        arg2 | tokenizer: transformers.PreTrainedTokenizer
            The transformer tokenizer to embed sequence of texts from a paper
        arg3 | embeddings_file[OPTIONAL, default=None]: pickle
            The pickled feature embeddings for a given paper's full-text
        arg4 | max_len[OPTIONAL, default=512]: int
            TBA
        arg5 | split_ratio[OPTIONAL, default=0.7]: float
            The Split ratio necessary for training, and testing the model
        ----------
        Returns
        -------
        Torch Dataset
            torch.utils.data.Dataset

        """
        # store the dataframe
        self.dataframe = dataframe

        # hf[tokenizer] initialization
        self.tokenizer = tokenizer

        # ocassionally store feature representations are passed from pickled files
        self.embeddings_file = embeddings_file

        # window size of the maximum
        self.max_len = max_len

        # dictionary to map embeddings, with doi
        self.embeddings_dict = None

        # check if an embedding file is passed
        if self.embeddings_file is not None:
            with open(self.embeddings_file, 'rb') as f:
                # load the pickled file
                self.embeddings_dict = pickle.load(f)

        # split ration to reserve sampels for training
        self.split_ratio = split_ratio

        # set training and test size
        train_size = int(len(self.dataframe) * self.split_ratio)
        test_size = len(self.dataframe) - train_size

        # prepare torch datasets
        self.train_ds, self.test_ds = random_split(self, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(2024))

    def __len__(self):
        return len(self.dataframe)

    def transform_labels(self, labels, zero_index=True):
        # zero index labels type check for long
        if zero_index:
            labels = labels - 1

        # type cast to long
        labels = labels.long()

        # return the one hot encoded targets
        return labels

    def train_dataloader(self, batch_size=32):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)

    def test_dataloader(self, batch_size=32):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    def __getitem__(self, idx):
        # get the paper
        paper = self.dataframe.iloc[idx]

        # get the parsed full text
        full_text = paper['s2_parse_full_text']

        # obtain paper id
        doi = paper['doi']

        # author centric spectrum target label
        auth_label = paper['auth_label']

        # external agent spectrum target label
        extagent_label = paper['extagent_label']

        if self.embeddings_dict is not None:
            # get the generated embeddings
            embeddings = self.embeddings_dict[doi]
        else:
            # tokenize the texts
            encoding = self.tokenizer.encode_plus(
                full_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # flatten the embedding feature matrix
            embeddings = encoding['input_ids'].flatten()

        # return the embedding and targets
        return {
            'input_ids': embeddings.unsqueeze(0),
            'attention_mask': encoding['attention_mask'].flatten().unsqueeze(0),
            'y_auth': torch.tensor(auth_label).unsqueeze(0),
            'y_ext_agent': torch.tensor(extagent_label).unsqueeze(0)
        }

# dataset class to build a torch dataloader with embeddings of papers
class ACMPaperTextDatasetFromDisk(Dataset):
    def __init__(self, embeddings_file, embedding_dict_key, split_ratio=0.7):
        """
        Helper class to load tensors, targets from disk
        for a given paper into a torch Dataset

        Parameters
        ----------
        arg1 | embeddings_file: pickle
            The pickled feature embeddings for a given paper's full-text
        arg2 | embedding_dict_key: str
            The key that holds the embedding for each paper
        arg3 | split_ratio[OPTIONAL, default=0.7]: float
            The Split ratio necessary for training, and testing the model
        ----------
        Returns
        -------
        Torch Dataset
            torch.utils.data.Dataset

        """

        # obtain feature representations are passed from pickled files
        self.embeddings_file = embeddings_file

        # dictionary to map embeddings, with doi
        self.embeddings_dict = None

        # key that holds the embedding
        self.embedding_dict_key = embedding_dict_key

        # check if an embedding file is passed
        if self.embeddings_file is not None:
            with open(self.embeddings_file, 'rb') as f:
                # load the pickled file
                self.embeddings_dict = pickle.load(f)

        # split ration to reserve sampels for training
        self.split_ratio = split_ratio

        # set training and test size
        train_size = int(len(self.embeddings_dict) * self.split_ratio)
        test_size = len(self.embeddings_dict) - train_size

        # prepare torch datasets
        self.train_ds, self.test_ds = random_split(self, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(2024))

    def __len__(self):
        return len(self.embeddings_dict)

    def transform_labels(self, labels, zero_index=True):
        # zero index labels type check for long
        if zero_index:
            labels = labels - 1

        # type cast to long
        labels = labels.long()

        # return the one hot encoded targets
        return labels

    def train_dataloader(self, batch_size=32):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)

    def test_dataloader(self, batch_size=32):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    def __getitem__(self, idx):
        # get the generated embeddings
        embeddings = self.embeddings_dict[idx][self.embedding_dict_key]
        # embeddings = embeddings.reshape(1, -1)

        # check if the input tensor must be averaged
        if embeddings.ndim > 1:
            # average over the encoder outputs
            embeddings = torch.mean(embeddings, dim=0)
        
        # ensure the embeddings are of float
        embeddings = embeddings.float()

        # author centric spectrum target label
        auth_label = self.embeddings_dict[idx]['auth_label']

        # external agent spectrum target label
        extagent_label = self.embeddings_dict[idx]['extagent_label']

        # return the embedding and targets
        return {
            'input_ids': embeddings,
            'y_auth': torch.tensor(auth_label, dtype=torch.long),
            'y_ext_agent': torch.tensor(extagent_label, dtype=torch.long)
        }

# function for processing the dataset
def data_processing(file_path):
    """
    Load reproducible papers CSV from a filepath and process features to
    prepare the dataframe for downstream tests, and modeling tasks

    Parameters
    ----------
    arg1 | file_path: str
        The file path indicating the location of the dataset

    Returns
    -------
    Dataframe
        pandas.DataFrame

    """
    try:
        # init scaling object for min maxing median readability
        mm = MinMaxScaler(feature_range=(0, 1))

        # read the dataframe
        raw_data = pd.read_csv(file_path)

        # change funding source column
        raw_data = raw_data.assign(fund_avail = list(map(lambda x: 1 if type(x) == str else 0, raw_data.funding_sources)))

        # change conf_repro_art_checklist column
        raw_data = raw_data.assign(conf_repro_art_checklist = list(map(lambda x: 1 if x == True else 0, raw_data.conf_repro_art_checklist)))

        # change conf_repro_art_eval_at_submission column
        raw_data = raw_data.assign(conf_repro_art_eval_at_submission = list(map(lambda x: 1 if x == True else 0, raw_data.conf_repro_art_eval_at_submission)))

        # change conf_repro_awards column
        raw_data = raw_data.assign(conf_repro_awards = list(map(lambda x: 1 if x == True else 0, raw_data.conf_repro_awards)))

        # change conf_repro_process_auth_comm column
        raw_data = raw_data.assign(conf_repro_process_auth_comm = list(map(lambda x: 1 if x == True else 0, raw_data.conf_repro_process_auth_comm)))

        # literal eval the badges list
        raw_data["badges"] = raw_data["badges"].apply(literal_eval)

        # get unique badges for every article
        raw_data["badges_set"] = raw_data["badges"].apply(set)

        # apply scaling on the readability
        raw_data.median_readability = mm.fit_transform(raw_data.median_readability.values.reshape(-1, 1))

        # return the dataframe
        return raw_data
    except:
        return pd.DataFrame()

# function for preparing feature list
def fetch_X_y_details():
    """
    Return the list of features(names) in consideration
    for downstream statistical tests, and predictive modeling tasks

    Parameters
    ----------
    None
    ----------
    Returns
    -------
    Tuple
        (Dict, Dict, List, List, Dict, Dict)

    """
    try:
        # gather the list of linguistic features
        linguistic_features = ["median_readability", "lexical_diversity_mtld"]

        # gather the list of scholarly features
        scholarly_features = ["gs_citations_23"]

        # gather the list of structural features
        structural_features = ["no_alg", "no_equations"]

        # gather venue specific reproducibility features
        venue_repro_features = ["conf_repro_art_checklist", "conf_repro_art_eval_at_submission", "conf_repro_awards", "conf_repro_process_auth_comm"]

        # gather artifact related features
        artifact_repro_features = ["zenodo_art_mention", "github_art_mention", "pwc_github_repo_mention", "pwc_datasets_mention", "pwc_methods_mention"]

        # gather misc features
        misc_features = ["fund_avail", "suppl_info"]

        # gather the target value
        badged_status = ["badged_reproducible"]

        # dictionary to store elaborate feature names
        feature_name_mapping = {
            "gs_citations_23": "Google Scholar citations | Y2023",
            "no_alg": "Number of Algorithms",
            "no_equations": "Number of Equations",
            "conf_repro_art_checklist": "Venue - Availability of Reproducibility checklist",
            "conf_repro_art_eval_at_submission": "Venue - Mandatory artifact submission for papers",
            "conf_repro_awards": "Venue - Reproducibility Awards",
            "conf_repro_process_auth_comm": "Venue - Author Correspondence for Reproducibility",
            "zenodo_art_mention": "Mention of Zenodo Artifacts",
            "github_art_mention": "Mention of GitHub Code Repository",
            "pwc_github_repo_mention": "Mention of Papers With Code GitHub Official Repository",
            "pwc_datasets_mention": "Mention of Papers With Code Datasets",
            "pwc_methods_mention": "Mention of Papers With Code Methods",
            "median_readability": "Median Readability",
            "lexical_diversity_mtld": "Measure of lexical textual diversity",
            "fund_avail": "Availability of Funding source",
            "suppl_info": "Availability of Supplemental information",
            "auth_prob_0": "$P(y= A_{PWA} | X_i)$",
            "auth_prob_1": "$P(y= A_{PUNX} | X_i)$",
            "auth_prob_2": "$P(y= A_{PAX} | X_i)$"
        }

        # list of numerical features useful for statistical tests
        numerical_features = ["median_readability", "no_alg", "no_equations", "gs_citations_23", "lexical_diversity_mtld"]

        # form a list of comprising of all the features
        X =  {"structural_features":structural_features,
              "scholarly_features": scholarly_features,
              "venue_repro_features": venue_repro_features,
              "artifact_repro_features": artifact_repro_features,
              "linguistic_features": linguistic_features,
              "misc_features": misc_features,
              "badged_status": badged_status}

        # list of other features necessary for downstream modeling and data analysis
        other_features = ["badges", "badges_set", "auth_label", "extagent_label", "joint_label"]

        # author centric framework dictionary
        author_centric_framework = {
            1: "Papers without artifacts",
            2: "Papers with artifacts that aren't permanantly archived",
            3: "Papers with artifacts that are permanantly archived"
        }

        # external agent framework for assessing reproducibility
        external_agent_framework = {
            0: "Papers that cannot be reproduced",
            1: "Paper Awaiting-Reproducibility",
            2: "Reproduced paper",
            3: "Reproducibile paper"
        }

        # return the triplet
        return X, feature_name_mapping, numerical_features, other_features, author_centric_framework, external_agent_framework
    except:
        return dict(), dict(), list(), list(), dict(), dict()

# function to obtain badged and unbadged articles
def prepare_data(badged_file_path, unbadged_file_path):
    """
    Main function to proces the data, & create torch dataloader objects
    for downstream statistical tests, and predictive modeling tasks

    Parameters
    ----------
    arg1 | badged_file_path: str
        The file path for badged papers dataframe
    arg2 | unbadged_file_path: str
        The file path for unbadged papers dataframe
    ----------
    Returns
    -------
    Tuple
        (Dict, Dict, List)

    """
    # call on fetch x, and y details function to get featuresn and target label names
    X, _, _, other_features, author_centric_framework, external_agent_framework = fetch_X_y_details()

    # init a scaler object for dataframe to torch transformation
    scaler = StandardScaler()

    # read the badged articles dataframe
    badged_articles = pd.read_csv(badged_file_path)

    # read the unbadged articles dataframe
    unbadged_articles = pd.read_csv(unbadged_file_path)

    # init empty list objects for label construction
    auth_framework_labels_badged, extagent_framework_labels_badged, joint_framework = [], [], []

    # extract auth centric labels on badged articles
    for paper in badged_articles.badges_set:
        # check if the paper has artifacts evaluated and functional
        if "Artifacts Evaluated & Functional" in ", ".join(literal_eval(paper)):
            auth_framework_labels_badged.append(2)
        else:
            auth_framework_labels_badged.append(3)

    # extract external agent centric labels on badged articles
    for paper in badged_articles.badges_set:
        # assign the paper badge
        paper_badge = ", ".join(literal_eval(paper))

        # check if the paper reproducibility level
        if "Results Reproduced" in paper_badge:
            # check if reproduced or reproducible
            if "Artifacts Evaluated" in paper_badge:
                # add reproducible label
                extagent_framework_labels_badged.append(3)

                # add highest standard satisfied for reproducibility label
                joint_framework.append(1)
            else:
                # add reproduced label
                extagent_framework_labels_badged.append(2)

                # add doesn't satisfy highest standard for reproducibility label
                joint_framework.append(0)
        else:
            # add awaiting reproducibility label
            extagent_framework_labels_badged.append(1)

            # add doesn't satisfy highest standard for reproducibility label
            joint_framework.append(0)

    # assign author centric framework labels
    badged_articles = badged_articles.assign(auth_label = auth_framework_labels_badged)

    # assign author centric framework labels on unbadged papers
    unbadged_articles = unbadged_articles.assign(auth_label = np.ones(unbadged_articles.shape[0], dtype=int))
    logger.info("Assigning Author centric labels......")

    # assign external agent centric framework labels
    badged_articles = badged_articles.assign(extagent_label = extagent_framework_labels_badged)

    # assign external agent centric framework labels on unbadged papers
    unbadged_articles = unbadged_articles.assign(extagent_label = np.zeros(unbadged_articles.shape[0], dtype=int))
    logger.info("Assigning External agent centric labels......")

    # assign joint framework label to all papers
    badged_articles = badged_articles.assign(joint_label = joint_framework)
    unbadged_articles = unbadged_articles.assign(joint_label = np.zeros(unbadged_articles.shape[0], dtype=int))
    logger.info("Assigning Joint framework labels......")

    # flatten the X list to only consider feature column names
    X = sum(X.values(), [])

    # exclude the badge_reproducible column
    X = X[:-1]

    # concat both the datasets to have a consolidated dataframe
    articles = pd.concat([badged_articles[X + other_features], unbadged_articles[X + other_features]]).reset_index(drop=False)

    # random shuffle the datadrame
    articles = articles.sample(frac=1, random_state=123)

    # transform the pandas dataframe to torch tensors
    badged_X = torch.tensor(badged_articles[X].values).float()
    unbadged_X = torch.tensor(unbadged_articles[X].values).float()
    articles_X = torch.tensor(articles[X].values).float()

    # transform the pandas dataframe by applying standard scaling to input features
    badged_scaled_X = torch.tensor(scaler.fit_transform(badged_articles[X].values)).float()
    unbadged_scaled_X = torch.tensor(scaler.fit_transform(unbadged_articles[X].values)).float()
    articles_scaled_X = torch.tensor(scaler.fit_transform(articles[X].values)).float()

    # type cast  the target variables on author centric and external agent centric labels
    badged_y_author_centric = torch.tensor(badged_articles["auth_label"].values).long()
    unbadged_y_author_centric = torch.tensor(unbadged_articles["auth_label"].values).long()
    articles_y_author_centric = torch.tensor(articles["auth_label"].values).long()
    badged_y_ext_agent_centric = torch.tensor(badged_articles["extagent_label"].values).long()
    unbadged_y_ext_agent_centric = torch.tensor(unbadged_articles["extagent_label"].values).long()
    articles_y_ext_agent_centric = torch.tensor(articles["extagent_label"].values).long()

    # type cast the target variable for joint framework
    badged_y_joint_label = torch.tensor(badged_articles["joint_label"].values).long()
    unbadged_y_joint_label = torch.tensor(unbadged_articles["joint_label"].values).long()
    articles_y_joint_label = torch.tensor(articles["joint_label"].values).long()

    # prepare the torch dataset
    badged_dataset = ACMPapersDataset(badged_X, badged_scaled_X, badged_y_author_centric, badged_y_ext_agent_centric, badged_y_joint_label)
    unbadged_dataset = ACMPapersDataset(unbadged_X, unbadged_scaled_X, unbadged_y_author_centric, unbadged_y_ext_agent_centric, unbadged_y_joint_label)
    articles_dataset = ACMPapersDataset(articles_X, articles_scaled_X, articles_y_author_centric, articles_y_ext_agent_centric, articles_y_joint_label)

    # prepare the torch data loader
    badged_loader = DataLoader(badged_dataset, batch_size=32, shuffle=True, num_workers=1,
                               worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(2024))
    unbadged_loader = DataLoader(unbadged_dataset, batch_size=32, shuffle=True, num_workers=1,
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(2024))
    articles_loader = DataLoader(articles_dataset, batch_size=32, shuffle=True, num_workers=1,
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(2024))
    logger.info("Preparing data loaders......")
    logger.info("------------------------------\n")

    # return badged and unbadged articles
    return badged_loader, badged_articles, unbadged_loader, unbadged_articles, articles_loader, articles

# function to obtain complete(badged and unbadged articles) paper-text, labels dataset
def prepare_paper_emb_data(paper_file_path, tokenizer):
    """
    Main function to process the dataframe with text, labels
    for predictive modeling tasks

    Parameters
    ----------
    arg1 | paper_file_path: str
        The file path for complete (badged, and unbadged papers) dataframe
    arg2 | tokenizer: str
        The tokenizer object to embed complete (badged, and unbadged papers) raw text
    ----------
    Returns
    -------
    DataLoader
        torch.data.DataLoader

    """
    # read the articles dataframe
    articles = pd.read_csv(paper_file_path)

    # random shuffle the datadrame
    articles = articles.sample(frac=1, random_state=123)

    # prepare the torch dataset
    articles_dataset = ACMPaperTextDataset(articles, tokenizer)

    # prepare the torch data loader
    articles_loader = DataLoader(articles_dataset, batch_size=32, shuffle=True, num_workers=1,
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(2024))

    logger.info("Preparing paper embedding data loaders......")
    logger.info("------------------------------\n")

    # return the articles dataloader
    return articles_loader