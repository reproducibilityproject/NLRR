import torch
import logging
import argparse
import numpy as np
from typing import Callable
from data.dataloader import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from transformers import LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer
from sklearn.metrics import accuracy_score, classification_report

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set the device appropriate for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# initialize longformer representations
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# helper function to train a representational learning model
def train_representations(
        model: torch.nn.Module,
        dataloader: DataLoader,
        target_label: str,
        one_hot_dim: int,
        zero_index: bool,
        iterations: int,
        optimizer: torch.optim.Optimizer,
        compute_loss: Callable,
        feature_name:str="scaled_X",
        logging:bool=True) -> torch.nn.Module:
    """
    Train and return the Pytorch model

    Parameters
    ----------
    arg1 | model: torch.nn.Module
        Trained Neural network model
    arg2 | dataloader: DataLoader
        Dataset as a DataLoader object
    arg3 | target_label: str
        Target Label for the nn model
    arg4 | one_hot_dim: int
        Dimensionality to consider before One hot transformation
    arg5 | zero_index: bool
        Zero the index of the target labels while performing one-hot transformation
    arg6 | iterations: int
        Iternations necessary for training the Neural network model
    arg7 | optimizer: torch.optim.Optimizer
        Torch optimizer for training a neural network model
    arg8 | compute_loss: Callable
        Loss criterion to calculate cost on y hats while training the Neural network model
    arg9 | logging[OPTIONAL]: bool
        Log the outputs of the training procedure
    Returns
    -------
    Pytorch model
        torch.nn.Module
    """
    for iteration in range(iterations):
        # set the model for training
        model.train()

        # iterate in batches over the training dataset
        for data in dataloader.dataset.train_dataloader():
            # set the gradients to zero
            optimizer.zero_grad()

            # forward pass and compute the y hat values
            y_hat = model(data[feature_name])

            # get true target
            y_true = dataloader.dataset.transform_labels(data[target_label], zero_index=zero_index)

            # compute the mean squared error loss
            cost = compute_loss(y_hat, y_true)

            # compute mse loss again for the backward pass
            cost.backward()

            # update the weights
            optimizer.step()

        # display the stats
        if logging:
            print(f'Epoch: {iteration:03d}, Loss: {cost:.4f}')

    # return the trained model
    return model

# helper function to train a joint model using logits
def train_joint_model_with_logits(
        joint_model: torch.nn.Module,
        dataloader: DataLoader,
        model_auth_centric: torch.nn.Module,
        model_ext_agent: torch.nn.Module,
        iterations: int,
        optimizer: torch.optim.Optimizer,
        compute_loss: Callable,
        logging: bool = False) -> torch.nn.Module:
    """
    Training loop for the joint[ACMnnJointLogits] model

    Parameters
    ----------
    arg1 | joint_model: torch.nn.Module
        Instantiated/Compiled Neural network model
    arg2 | dataloader: DataLoader
        Dataset as a DataLoader object
    arg3 | model_auth_centric: torch.nn.Module
        Fully trained ACMnn model for author centric labels
    arg4 | model_ext_agent: torch.nn.Module
        Fully trained ACMnn model for external agent labels
    arg5 | iterations: int
        Iternations necessary for training the Neural network model
    arg6 | optimizer: torch.optim.Optimizer
        Torch optimizer for training a neural network model
    arg8 | compute_loss: Callable
        Loss criterion to calculate cost on y hats while training the Neural network model
    arg9 | logging[OPTIONAL]: bool
        Log the outputs of the training procedure
    Returns
    -------
    Pytorch model
        torch.nn.Module
    """
    for iteration in range(iterations):
        # set the model for training
        joint_model.train()

        for data in dataloader:
            # set the gradients to zero
            optimizer.zero_grad()

            # obtain logits from both(author-centric, external-agent) models
            logits_auth = model_auth_centric(data['scaled_X'])
            logits_ext = model_ext_agent(data['scaled_X'])

            # concat and combine the logits
            combined_logits = torch.cat([logits_auth, logits_ext], dim=1)

            # get y hats for the joint model
            outputs = joint_model(combined_logits)

            # Assuming labels are stored in 'y' in the dataloader
            cost = compute_loss(outputs, data['y_joint'])

            # compute mse loss again for the backward pass
            cost.backward()

            # update the weights
            optimizer.step()

            # display the stats
            if logging:
                print(f'ITER: {iteration + 1}, Batch Loss: {cost.item():.4f}')

    # return the trained model
    return joint_model

# helper function to evaluate a joint model using logits
def evaluate_joint_model_with_logits(
        joint_model: torch.nn.Module,
        model_auth_centric: torch.nn.Module,
        model_ext_agent: torch.nn.Module,
        dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the Pytorch model and return
    ground truth along with predictions
    Parameters
    ----------
    arg1 | model: torch.nn.Module
        Trained Neural network model
    arg2 | dataloader: DataLoader
        Dataset as a DataLoader object
    arg3 | target_label: str
        Target Label for the nn model
    arg4 | reindex_targets: bool
        Reindex by adding one to the target labels inverse one-hot transformation
    """
    # init an empty list to capture y hats
    y_preds = []

    # init an empty list to capture ground truth
    y_true = []

    # set the model to evaluate
    joint_model.eval()

    with torch.no_grad():
        # Iterate in batches over the test dataset.
        for data in dataloader.dataset.test_dataloader():
            # obtain logits from both(author-centric, external-agent) models
            logits_auth = model_auth_centric(data['scaled_X'])
            logits_ext = model_ext_agent(data['scaled_X'])

            # concat and combine the logits
            combined_logits = torch.cat([logits_auth, logits_ext], dim=1)

            # store the ground truth
            y_true.append(data['y_joint'].numpy())

            # gather the model prediction
            out = joint_model(combined_logits)

            # store the model predictions
            y_preds.append(torch.argmax(out, dim=1).numpy())

    # concat the predictions obtained in batches
    # y_preds = torch.cat(y_preds)
    y_preds = np.concatenate(y_preds)

    # concat the ground truth obtained in batches
    # y_true = torch.cat(y_true)
    y_true = np.concatenate(y_true)

    # return the tuple [Ground truth, Predictions]
    return y_true, y_preds

# helper function to evaluate a representational learning model
def evaluate_representations(model: torch.nn.Module,
                             dataloader: DataLoader,
                             target_label: str,
                             reindex_targets: bool,
                             feature_name:str = "scaled_X",) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the Pytorch model and return
    ground truth along with predictions
    Parameters
    ----------
    arg1 | model: torch.nn.Module
        Trained Neural network model
    arg2 | dataloader: DataLoader
        Dataset as a DataLoader object
    arg3 | target_label: str
        Target Label for the nn model
    arg4 | reindex_targets: bool
        Reindex by adding one to the target labels inverse one-hot transformation
    """
    # init an empty list to capture y hats
    y_preds = []

    # init an empty list to capture ground truth
    y_true = []

    # set the model to evaluate
    model.eval()

    with torch.no_grad():
        # Iterate in batches over the test dataset.
        for data in dataloader.dataset.test_dataloader():
            # store the ground truth
            y_true.append(data[target_label].numpy())

            # gather the model prediction
            out = model(data[feature_name])

            # store the model predictions
            y_preds.append(torch.argmax(out, dim=1).numpy())

    # concat the predictions obtained in batches
    # y_preds = torch.cat(y_preds)
    y_preds = np.concatenate(y_preds)

    # concat the ground truth obtained in batches
    # y_true = torch.cat(y_true)
    y_true = np.concatenate(y_true)

    # check if re-indexing the zero index array is necessary
    if reindex_targets:
        # add by one
        y_preds += 1

    # return the tuple [Ground truth, Predictions]
    return y_true, y_preds

# VanillaNN deep learning model
class ACMnn(torch.nn.Module):
    def __init__(self, ip_dim, op_dim):
        """
        Initializes the Torch Vanilla NN model with the given parameters.

        Parameters
        ----------
        arg1 | ip_dim (int)
            The dimensionality of input feature representations.
        arg2 | op_dim (int)
            The number of targets to predict.
        ----------

        Returns
        -------
        Model
            torch.nn.Module

        """
        # set a manual seed
        torch.manual_seed(2024)

        # temporary object of the ACMnn
        super(ACMnn, self).__init__()

        # feature representations
        self.inp = torch.nn.Linear(ip_dim, 256)

        # hiddent representations
        self.hidden = torch.nn.Linear(256, 128)

        # output representations
        self.op = torch.nn.Linear(128, op_dim)

    def forward(self, x):
        # non linear activations on representations
        x = torch.relu(self.inp(x))
        x = torch.relu(self.hidden(x))
        x = self.op(x)

        # apply softmax and return outputs
        return x

# Simple representational deep learning model
class ACMSimpleModel(torch.nn.Module):
    def __init__(self, ip_dim, op_dim):
        """
        Initializes the Torch ACMSimpleModel NN model with the given parameters.

        Parameters
        ----------
        arg1 | ip_dim (int)
            The dimensionality of input feature representations.
        arg2 | op_dim (int)
            The number of targets to predict.
        ----------

        Returns
        -------
        Model
            torch.nn.Module

        """
        # set a manual seed
        torch.manual_seed(2024)

        # temporary object of the ACMSimpleModel
        super(ACMSimpleModel, self).__init__()

        # output representations
        self.clf = torch.nn.Linear(ip_dim, op_dim)

    def forward(self, x):
        # obtain output activations on representations
        out = self.clf(x)

        # apply softmax and return outputs
        return out

# Vanill nn model that utilizes joint logits
class ACMnnJointLogits(torch.nn.Module):
    def __init__(self, ip_dim, op_dim):
        """
        Initializes the ACMnnJointLogits model with the given parameters.

        Parameters
        ----------
        arg1 | ip_dim (int)
            The dimensionality of input feature representations.
        arg2 | op_dim (int)
            The number of targets to predict.
        ----------

        Returns
        -------
        Model
            torch.nn.Module

        """
        # set a manual seed
        torch.manual_seed(2024)

        # temporary object of the ACMnnJointLogits
        super(ACMnnJointLogits, self).__init__()

        # feature representations in the form of logits
        self.fc1 = torch.nn.Linear(ip_dim, 128)

        # output representations
        self.fc2 = torch.nn.Linear(128, op_dim)

    def forward(self, x):
        # non linear activations on logits from previous models
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # apply softmax and return outputs
        return torch.softmax(x, dim=1)

# helper module for scalar mixing weights in an attention based model
class ACMnnAttnScalarMix(torch.nn.Module):
    def __init__(self, layers):
        """
        Initializes the Torch Scalar weight mixing model.

        Parameters
        ----------
        arg1 | layers (int)
            The number of layers to gather weights.
        ----------

        Returns
        -------
        Model
            torch.nn.Module

        """
        # temporary object of ACMnnAttnScalarMix class
        super(ACMnnAttnScalarMix, self).__init__()

        # gather the weights
        self.weights = torch.nn.Parameter(torch.zeros(layers))

    def forward(self, inputs, dim=0):
        # apply softmax over the weights
        weights = torch.softmax(self.weights, dim=0)

        # prepare for output pooling
        mixed_inp = torch.zeros_like(inputs[0])

        # iterate to pool inputs
        for i, input in enumerate(inputs):
            mixed_inp += weights[i] * input

        # return scalar mixed inputs
        return mixed_inp

# attention based representational learning model
class ACMnnAttnEmb(torch.nn.Module):
    def __init__(self, op_dim, num_layers=12, attention_window=512):
        """
        Initializes the Torch Attnetion based model scalar mixing model.

        Parameters
        ----------
        arg1 | op_dim (int)
            The number of targets to predict.
        arg2 | num_layers (int)
            The number of layers for the scalar mixing.
        arg3 | attention_window (int)
            The size of the attention window.
        ----------

        Returns
        -------
        Model
            torch.nn.Module

        """
        # set a manual seed
        torch.manual_seed(2024)

        # temporary object of the ACMnn
        super(ACMnnAttnEmb, self).__init__()

        # long former model config
        acm_longformer_config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        acm_longformer_config.attention_window = [attention_window] * acm_longformer_config.num_hidden_layers

        # input feature transformer representations
        self.inp = LongformerForSequenceClassification(acm_longformer_config)

        # scalar mixing outputs
        self.scalar_weights = ACMnnAttnScalarMix(num_layers)

        # apply layer norm for stablization and convergence
        self.layer_norm = torch.nn.LayerNorm(2)

        # hiddent representations
        self.hidden = torch.nn.Linear(2, 128)

        # dropout
        self.dropout = torch.nn.Dropout(0.3)

        # output representations
        self.op = torch.nn.Linear(128, op_dim)

    def forward(self, x_ids, attention_mask, global_attention_mask=None):
        # squash extra dimension from batch
        x_ids = x_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        global_attention_mask = global_attention_mask.squeeze(1)

        # get embedding outputs from transformer model
        ops = self.inp(input_ids=x_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

        # get pooled output logits
        longformer_ops = ops.logits

        # scalar mixing on ops
        mixed_ops = self.scalar_weights([longformer_ops] * 12)

        # apply layer normalization
        layer_norm_ops = self.layer_norm(mixed_ops)

        # apply non linear activation on scalar outputs
        hidden_ops = torch.relu(self.hidden(layer_norm_ops))

        # apply dropout
        dropout_ops = self.dropout(hidden_ops)

        # prepare outputs
        out = self.op(dropout_ops)

        # return outputs
        return out

# tree based supervised learning model
class TreeBasedModel:
    def __init__(self, choice="logistic"):
        """
        Initializes the RandomForest model with the given parameters.

        Parameters
        ----------
        arg1 | n_estimators (int)
            The number of trees in the forest.
        arg2 | max_depth (int)
            The maximum depth of the tree.
        ----------

        Returns
        -------
        Model
            sklearn.ensemble.RandomForestClassifier

        """
        # alter the model type based on choice
        if choice == "adb":
            # ensemble boosting classifier
            self.model = AdaBoostClassifier(n_estimators=100, random_state=2024)
        elif choice == "gdb":
            # ensemble classifier
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=2024)
        elif choice == "dt":
            # tree based classifier
            self.model = DecisionTreeClassifier(criterion="entropy", random_state=2024)
        elif choice == "rf":
            # random forest classifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=2024)
        else:
            # default model choice
            self.model = LogisticRegression(solver="lbfgs", random_state=2024)

    def fit(self, X, y):
        """
        Fits the treebased(default=LogisticRegression) model on non-normalized/unscaled data.

        Parameters
        ----------
        arg1 | X (np.array or pd.DataFrame)
            Training data.
        arg2 | y (np.array or pd.Series)
            Labels.
        ----------

        Returns
        -------
        Nothing

        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Makes predictions using the RandomForest model.

        Parameters
        ----------
        arg1 | X (np.array or pd.DataFrame)
            Data for making predictions.

        -------
        Returns
        -------
        Array
            np.array

        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluates the accuracy of the model.

        Parameters
        ----------
        arg1 | X (np.array or pd.DataFrame)
            Data for evaluation.
        arg2 | y (np.array or pd.Series)
            True labels.

        -------
        Returns
        -------
        Decimal
            float
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# function to train ML/DL model on acm badged articles
def run_training_inference_pipeline(args):
    """
    Main function to process the data, train models & provide eval metrics

    Parameters
    ----------
    arg1 | args: argparse.ArgumentParser
        The command line arguments necessary to run the main function
    ----------
    Returns
    -------
    Dataframe
        pandas.data.DataFrame

    """
    # ignore individual dataloaders and fetch concatenated dataloader
    _, _, _, _, articles_loader, _ = prepare_data(
        badged_file_path=args.badged_file_path,
        unbadged_file_path=args.unbadged_file_path
    )

    # obtain feature label information
    _, _, _, _, author_centric_framework, external_agent_framework = fetch_X_y_details()

    # Train models for learning author centric and external agent framework labels
    if args.train_model == 'tree_based':
        logger.info("Training Tree Based Model...")
        logger.info(args.tree_based_model)
        logger.info("------------------------------")

        # init the tree based model
        tree_model_auth_centric = TreeBasedModel(choice=args.tree_based_model)
        tree_model_ext_agent = TreeBasedModel(choice=args.tree_based_model)

        # transform features and targets to NumPy arrays for scikit-learn compatibility
        X_train, y_train_auth, y_train_ext_agent = [], [], []
        X_test, y_test_auth, y_test_ext_agent = [], [], []

        # prepare training data
        for data in articles_loader.dataset.train_dataloader():
            X_train.append(data['X'].numpy())
            y_train_auth.append(data['y_auth'].numpy())
            y_train_ext_agent.append(data['y_ext_agent'].numpy())

        # prepare training data
        for data in articles_loader.dataset.test_dataloader():
            X_test.append(data['X'].numpy())
            y_test_auth.append(data['y_auth'].numpy())
            y_test_ext_agent.append(data['y_ext_agent'].numpy())

        # transform the feature representations by horizontal stacking
        X_train, X_test = np.vstack(X_train), np.vstack(X_test)

        # transform the targets by horizontal flattening
        y_train_auth, y_train_ext_agent = np.concatenate(y_train_auth), np.concatenate(y_train_ext_agent)
        y_test_auth, y_test_ext_agent = np.concatenate(y_test_auth), np.concatenate(y_test_ext_agent)

        # Train the model
        tree_model_auth_centric.fit(X_train, y_train_auth)
        tree_model_ext_agent.fit(X_train, y_train_ext_agent)

        # evaluate the model
        y_hat_auth_centric = tree_model_auth_centric.predict(X_test)
        y_hat_ext_agent = tree_model_ext_agent.predict(X_test)

        logger.info("Tree Model CLF report for author centric framework")
        logger.info('Test Acc for the CLF model: ' + str(accuracy_score(y_hat_auth_centric, y_test_auth)))
        print(classification_report(y_test_auth, y_hat_auth_centric, target_names=list(author_centric_framework.values()), zero_division=1))
        logger.info("------------------------------")

        logger.info("Tree Model CLF report for external agent labels")
        logger.info('Test Acc for the CLF model: ' + str(accuracy_score(y_hat_ext_agent, y_test_ext_agent)))
        print(classification_report(y_test_ext_agent, y_hat_ext_agent, target_names=list(external_agent_framework.values()), zero_division=1))
        logger.info("------------------------------")

    if args.train_model == 'vanilla':
        logger.info("Training ACMnn Vanilla Model...")
        logger.info("------------------------------")

        # create the model
        model_auth_centric = ACMnn(16, 3)
        model_ext_agent = ACMnn(16, 4)

        # set cost function and optimizer
        num_epochs = 50
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_auth_centric = torch.optim.Adam(model_auth_centric.parameters(), lr=0.001)
        optimizer_ext_agent = torch.optim.Adam(model_ext_agent.parameters(), lr=0.001)

        # train the author centric model
        model_auth_centric = train_representations(model_auth_centric, articles_loader, 'y_auth', 3, True, num_epochs, optimizer_auth_centric, criterion)

        # save the author centric model
        torch.save(model_auth_centric.state_dict(), './models/model_auth_centric.pth')
        logger.info("Saved author-centric model.")

        # train the external agent framework model
        model_ext_agent = train_representations(model_ext_agent, articles_loader, 'y_ext_agent', 4, False, num_epochs, optimizer_ext_agent, criterion)

        # save external agent framework model
        torch.save(model_ext_agent.state_dict(), './models/model_ext_agent.pth')
        logger.info("Saved external-agent model.")

        # evaluate the author centric model
        y_test_auth, y_hat_auth_centric = evaluate_representations(model_auth_centric, articles_loader, 'y_auth', True)

        # evaluate the external agent framework model
        y_test_ext_agent, y_hat_ext_agent = evaluate_representations(model_ext_agent, articles_loader, 'y_ext_agent', False)

        # display the outputs
        logger.info("ACMnn Vanilla Model CLF report for author centric framework")
        print(classification_report(y_test_auth, y_hat_auth_centric, target_names=list(author_centric_framework.values()), zero_division=1))
        logger.info("------------------------------")

        logger.info("ACMnn Vanilla Model CLF report for external agent labels")
        print(classification_report(y_test_ext_agent, y_hat_ext_agent, target_names=list(external_agent_framework.values()), zero_division=1))
        logger.info("------------------------------")

    # building a surrogate model
    if args.train_model == "surrogate":
        logger.info("Training Surrogate F_E ACMnn Model...")
        logger.info("------------------------------")

        # create the model
        model_auth_centric = ACMnn(16, 3)
        model_ext_agent = ACMnn(3, 4)

        # set cost function and optimizer
        num_epochs = 20
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_auth_centric = torch.optim.Adam(model_auth_centric.parameters(), lr=0.001)
        optimizer_ext_agent = torch.optim.Adam(model_ext_agent.parameters(), lr=0.001)

        # train the author centric model
        model_auth_centric = train_representations(model_auth_centric, articles_loader, 'y_auth', 3, True, num_epochs, optimizer_auth_centric, criterion)

        # evaluate the author centric model
        y_test_auth, y_hat_auth_centric = evaluate_representations(model_auth_centric, articles_loader, 'y_auth', True)

        # over n epochs
        for _ in range(num_epochs):
            # set the model to training
            model_ext_agent.train()

            # iterate over the training data loader
            for data in articles_loader.dataset.train_dataloader():

                # forward pass, backprop, weight updates
                optimizer_ext_agent.zero_grad()
                y_hat_ext_agent = model_ext_agent(model_auth_centric(data['scaled_X']))
                loss_ext_agent = criterion(y_hat_ext_agent, articles_loader.dataset.to_onehot(data['y_ext_agent'], 4, zero_index=False))
                loss_ext_agent.backward()
                optimizer_ext_agent.step()

        # evaluate the model
        model_ext_agent.eval()

        # prepare arrays for testing
        y_test_ext_agent, y_hat_ext_agent = [], []

        with torch.no_grad():
            # iterate over the evaluation data
            for data in articles_loader.dataset.test_dataloader():
                # get the prediction for author centric and ext agent labels
                out_auth_centric = model_auth_centric(data['scaled_X'])
                out_ext_agent = model_ext_agent(out_auth_centric)

                # append the predictions
                y_hat_ext_agent.append(torch.argmax(out_ext_agent, dim=1).numpy())

                # ground truth
                y_test_ext_agent.append(data['y_ext_agent'].numpy())

        # concat and prepare ground truths, and y_hats for evaluation
        y_test_ext_agent, y_hat_ext_agent = np.concatenate(y_test_ext_agent), np.concatenate(y_hat_ext_agent)

        # display the outputs
        logger.info("Surrogate F_E ACMnn Model CLF report for external agent labels")
        print(classification_report(y_test_ext_agent, y_hat_ext_agent, target_names=list(external_agent_framework.values()), zero_division=1))
        logger.info("------------------------------")

    # building a joint logit based model
    if args.train_model == "joint_model_with_logits":
        logger.info("Training Joint ACMnnJointLogits Model...")
        logger.info("------------------------------")

        # Load pre-trained models or instantiate them
        model_auth_centric = ACMnn(ip_dim=16, op_dim=3)
        model_ext_agent = ACMnn(ip_dim=16, op_dim=4)
        joint_model = ACMnnJointLogits(ip_dim=7, op_dim=2)

        # load weights of trained author centric model
        model_auth_centric.load_state_dict(torch.load("./models/model_auth_centric.pth"))
        logger.info("Loaded pre-trained author-centric model.")

        # load weights of trained external agent model
        model_ext_agent.load_state_dict(torch.load("./models/model_ext_agent.pth"))
        logger.info("Loaded pre-trained external-agent model.")

        # set cost function and optimizer
        num_epochs = 20
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_joint_model = torch.optim.Adam(joint_model.parameters(), lr=0.001)

        # external agent framework for assessing reproducibility
        joint_framework = {
            0: "$A_{PAX} \cap E_{R}$: Archived & Verified Reproducible",
            1: "Other papers",
        }

        # train the joint mode with probabilities
        model_pax_er = train_joint_model_with_logits(
            joint_model=joint_model,
            dataloader=articles_loader,
            model_auth_centric=model_auth_centric,
            model_ext_agent=model_ext_agent,
            optimizer=optimizer_joint_model,
            compute_loss=criterion,
            iterations=num_epochs,
            logging=False
        )

        # evaluate the joint framework model
        y_test, y_hat = evaluate_joint_model_with_logits(joint_model=model_pax_er,
                                                         model_auth_centric=model_auth_centric,
                                                         model_ext_agent=model_ext_agent,
                                                         dataloader=articles_loader)

        # display the outputs
        logger.info("Joint ACMnnJointLogits Model CLF report for joint framework")
        print(classification_report(y_test, y_hat, target_names=list(joint_framework.values()), zero_division=1))
        logger.info("------------------------------")
    
    # building a representational learning model using SPECTER@V2
    if args.train_emb_model == "nn_x_specter":
        logger.info("Training Representational Learning Model ACMSimpleModel() using SPECTER embeddings...")
        logger.info("------------------------------")

        # load the tensors from disk
        # articles_ds = ACMPaperTextDatasetFromDisk("./data/papers_with_sp_vec.pickle", "sp_vec")
        # articles_ds = ACMPaperTextDatasetFromDisk("./data/papers_with_ada_vec.pickle", "ada_vec")
        articles_ds = ACMPaperTextDatasetFromDisk("./data/papers_with_longformer_vec.pickle", "longformer_vec")

        # prepare the torch data loader
        articles_loader = DataLoader(articles_ds, batch_size=32, shuffle=True, num_workers=1,
                                     worker_init_fn=seed_worker, 
                                     generator=torch.Generator().manual_seed(2024))

        logger.info("Prepared torch loader with specter embedding data loaders......")
        logger.info("------------------------------\n")

        # create the model
        # model_auth_centric = ACMSimpleModel(768, 3).to(device)
        model_auth_centric = ACMnn(768, 3).to(device)
        # model_auth_centric = ACMSimpleModel(1536, 3).to(device)
        # model_auth_centric = ACMnn(1536, 3).to(device)
        # model_ext_agent = ACMSimpleModel(768, 4).to(device)
        model_ext_agent = ACMnn(768, 4).to(device)
        # model_ext_agent = ACMSimpleModel(1536, 4).to(device)
        # model_ext_agent = ACMnn(1536, 4).to(device)

        # set cost function and optimizer
        num_epochs = 50
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_auth_centric = torch.optim.Adam(model_auth_centric.parameters(), lr=0.001)
        optimizer_ext_agent = torch.optim.Adam(model_ext_agent.parameters(), lr=0.001)

        # train the author centric model
        model_auth_centric = train_representations(model_auth_centric, articles_loader, 'y_auth', 3, \
                                                   True, num_epochs, optimizer_auth_centric, criterion,
                                                   feature_name="input_ids")

        # # save the author centric model
        # torch.save(model_auth_centric.state_dict(), './models/model_auth_centric.pth')
        # logger.info("Saved author-centric model.")

        # train the external agent framework model
        model_ext_agent = train_representations(model_ext_agent, articles_loader, 'y_ext_agent', 4, \
                                                False, num_epochs, optimizer_ext_agent, criterion,
                                                feature_name="input_ids")

        # # save external agent framework model
        # torch.save(model_ext_agent.state_dict(), './models/model_ext_agent.pth')
        # logger.info("Saved external-agent model.")

        # evaluate the author centric model
        y_test_auth, y_hat_auth_centric = evaluate_representations(model_auth_centric, articles_loader, \
                                                                   'y_auth', True, feature_name="input_ids")

        # evaluate the external agent framework model
        y_test_ext_agent, y_hat_ext_agent = evaluate_representations(model_ext_agent, articles_loader, \
                                                                     'y_ext_agent', False, feature_name="input_ids")

        # display the outputs
        logger.info("ACMSimpleModel Representational Learning Model CLF report for author centric framework")
        print(classification_report(y_test_auth, y_hat_auth_centric, target_names=list(author_centric_framework.values()), zero_division=1))
        logger.info("------------------------------")

        logger.info("ACMSimpleModel Representational Learning Model CLF report for external agent labels")
        print(classification_report(y_test_ext_agent, y_hat_ext_agent, target_names=list(external_agent_framework.values()), zero_division=1))
        logger.info("------------------------------")

    # building a paper representational learning model using Longformer
    if args.train_emb_model == "nn_x_longformer":
        logger.info("Training Paper Representational Learning Model ACMnnAttnEmb()...")
        logger.info("------------------------------")

        # ignore individual dataloaders and fetch concatenated dataloader
        articles_loader = prepare_paper_emb_data(args.paper_file_path, tokenizer)

        # create the model
        model_auth_centric = ACMnnAttnEmb(3).to(device)
        # model_ext_agent = ACMnnAttnEmb(4).to(device)

        # set cost function and optimizer
        num_epochs = 20
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_auth_centric = torch.optim.Adam(model_auth_centric.parameters(), lr=0.001)
        # optimizer_ext_agent = torch.optim.Adam(model_ext_agent.parameters(), lr=0.001)

        # over n epochs
        for epoch in range(num_epochs):
            # set the model to training
            model_auth_centric.train()

            # iterate over the training data loader
            for iteration, data in enumerate(articles_loader.dataset.train_dataloader()):
                # get the inputs and attention mask
                inp_ids = data['input_ids'].to(device)
                attn_mask = data['attention_mask'].to(device)

                # setup global attention mask
                global_attn_mask = torch.zeros_like(attn_mask)

                # apply the global attention on primary token
                global_attn_mask[:, 0] = 1

                # get the ground truth
                y_true = data['y_auth'].to(device).squeeze(1) - 1

                # forward pass, backprop, weight updates
                optimizer_auth_centric.zero_grad()

                # get y hat for the batch
                y_hat_auth_agent = model_auth_centric(inp_ids, attn_mask, global_attn_mask)

                # compute cost of the pass
                loss_auth_agent = criterion(y_hat_auth_agent, y_true)

                # update the cost function
                loss_auth_agent.backward()

                # apply weights
                optimizer_auth_centric.step()

                # get the predicted labels for capturing the training eval metrics
                predicted_labels = torch.argmax(y_hat_auth_agent, dim=1)
                accuracy = (predicted_labels == y_true).float().mean().item()

                print(f'Epoch: {epoch + 1:03d}, Iteration: {iteration:04d}, Loss: {loss_auth_agent:.4f}, Train Acc: {accuracy}')

if __name__ == '__main__':
    # create an argparse object
    parser = argparse.ArgumentParser(description='src/learning.py: Train ML/DL mdoels on Scholarly papers to predict author centric labels and external agent labels for reproducibility.')

    # itemize all of the available command line arguments
    parser.add_argument('--badged_file_path', type=str, default='./data/papers_with_badged_information.csv',
                help='File path to the badged papers data CSV file.')
    parser.add_argument('--unbadged_file_path', type=str, default='./data/papers_without_badged_information.csv',
                help='File path to the unbadged papers data CSV file.')
    parser.add_argument('--paper_file_path', type=str, default='./data/acmpapers_fulltext_labels_04_24.csv',
                help='File path to the raw papers text-labels dataframe file.')
    parser.add_argument('--train_model', type=str, choices=['tree_based', 'vanilla', 'surrogate', 'joint_model_with_logits', 'all'], default='all',
                help='Specify which learning model to train: Tree Based Model, VanillaNN, or all.')
    parser.add_argument('--tree_based_model', type=str, choices=["adb", "gdb", "dt", "rf", "log"], default='gdb',
                help='Specify which tree-based ML learning algorithm to train: AdaBoost, Gradient Boosting, Decision Tree, or Random Forest.')
    parser.add_argument('--train_emb_model', type=str, choices=['nn_x_ada', 'nn_x_llama3', 'nn_x_specter', 'nn_x_longformer', 'none'], default='none',
                help='Specify which embedding representations to train and learn : NN with X(ADA), NN with X(LlaMa3), or all.')
    parser.add_argument('--feature_type', type=str, choices=['normal', 'scaled'], default='normal',
                help='Choose the type of features to use: Normal or Scaled.')

    # parse the arguments from the given choices
    args = parser.parse_args()

    # run the model
    run_training_inference_pipeline(args)
