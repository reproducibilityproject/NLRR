import logging
import argparse
from data.dataloader import *
from scipy.stats import f_oneway, kruskal, shapiro, levene

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# helper function to run [ANOVA, Kruskal, Shapiro, Levene] test
def perform_test(articles, target, numerical_features, feature_name_mapping, test_type):
    """
    Helper function to utilize the articles data, & run ANOVA test

    Parameters
    ----------
    arg1 | articles: pandas.data.DataFrame
        The dataframe with numerical features necessary for performing the experiment
    arg2 | target: torch.Tensor
        The target variable in question for determining the groups of ANOVA test
    arg3 | numerical_features: list
        List of column names used to identify numerical features and map them from features tensor
    arg4 | feature_name_mapping: list
        List of expanded column names for the input numerical features
    arg4 | test_type: str
        Type of statistical test to perform
    ----------
    Returns
    -------
    Dataframe
        pandas.data.DataFrame

    """
    try:
        # init dict to store results
        results = {}

        # do one-way ANOVA
        for feature in numerical_features:
            # get author centric fraemwork groups
            groups = [articles[articles[target] == label][feature] for label in articles[target].unique()]

            # init statistic, and p-value
            stat, p_value = None, None

            # check the type of test
            if test_type == "anova":
                # one-way anova
                stat, p_value = f_oneway(*groups)

                # store the results
                results[feature_name_mapping[feature]] = {'F-statistic': stat, 'p-value': p_value}
            if test_type == "kruskal":
                # run kruskal-wallis test
                stat, p_value = kruskal(*groups)

                # store the results
                results[feature_name_mapping[feature]] = {'Statistic': stat, 'p-value': p_value}
            if test_type == "shapiro":
                # run shapiro-wilk test for normality of distribution
                stat, p_value = shapiro(articles[feature])

                # store the results
                results[feature_name_mapping[feature]] = {'Statistic': stat, 'p-value': p_value}
            if test_type == "levene":
                # run levene's test for equity of variance
                stat, p_value = levene(*groups)

                # store the results
                results[feature_name_mapping[feature]] = {'Statistic': stat, 'p-value': p_value}

        # store the resuls in df
        logger.info("Grouped by " + target + " framework")
        results_df = pd.DataFrame(results).T

        # return the results dataframe
        return results_df
    except Exception as e:
        # throw the exception
        logger.error("ERR: Unable to run " + test_type + " test on the data: ", str(e))

        # return empty dataframe
        return pd.DataFrame()

# function to run statistical tests on acm badged articles
def run_statistical_analysis(args):
    """
    Main function to process the data, & run statistical tests

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
    # obtain both the dataloaders
    _, _, _, _, _, articles = prepare_data(
        badged_file_path=args.badged_file_path,
        unbadged_file_path=args.unbadged_file_path
    )

    # obtain feature label information
    _, feature_name_mapping, numerical_features, _, _, _ = fetch_X_y_details()

    # Run tests for normality and homogeniety of variances
    if args.run_test == 'all' or args.run_test == 'shapiro':
        logger.info("Running Shapiro-Wilk Test for normality...")

        # log the results
        logger.info("------------------------------")
        logger.info("Shapiro-Wilk Test for normality results on scholarly papers")
        
        # call Shapiro helper function to test author centric framework
        shapiro_auth_centric_framework = perform_test(articles, 'auth_label', numerical_features, feature_name_mapping, 'shapiro')

        print(shapiro_auth_centric_framework)
        logger.info("------------------------------")
        logger.info("Shapiro-Wilk Test for normality results on scholarly papers")

        # call Shapiro helper function to test external agent centric framework
        shapiro_ext_agent_framework = perform_test(articles, 'extagent_label', numerical_features, feature_name_mapping, 'shapiro')

        print(shapiro_ext_agent_framework)
        logger.info("------------------------------")

    if args.run_test == 'all' or args.run_test == 'levene':
        logger.info("Running Levene's Test for homogeneity of variances...")

        # log the results
        logger.info("------------------------------")
        logger.info("Levene's Test for homogeneity of variances on scholarly papers")

        # call Levene's Test helper function to test author centric framework
        levene_auth_centric_framework = perform_test(articles, 'auth_label', numerical_features, feature_name_mapping, 'levene')

        print(levene_auth_centric_framework)
        logger.info("------------------------------")
        logger.info("Levene's Test for homogeneity of variances on scholarly papers")

        # call Levene's Test helper function to test external agent centric framework
        levene_ext_agent_framework = perform_test(articles, 'extagent_label', numerical_features, feature_name_mapping, 'levene')

        print(levene_ext_agent_framework)
        logger.info("------------------------------")

    # Perform statistical tests based on the command line arguments
    if args.run_test == 'all' or args.run_test == 'anova':
        logger.info("Running ANOVA...")

        # log the results
        logger.info("------------------------------")
        logger.info("ANOVA results for scholarly papers grouped by author centric framework")

        # call anova helper function to test author centric framework
        anova_auth_centric_framework = perform_test(articles, 'auth_label', numerical_features, feature_name_mapping, 'anova')
        print(anova_auth_centric_framework)
        logger.info("------------------------------")
        
        logger.info("ANOVA results for scholarly papers grouped by external agent framework")

        # call anova helper function to test external agent centric framework
        anova_ext_agent_framework = perform_test(articles, 'extagent_label', numerical_features, feature_name_mapping, 'anova')
        print(anova_ext_agent_framework)
        logger.info("------------------------------")

    if args.run_test == 'all' or args.run_test == 'kruskal':
        logger.info("Running Kruskal-Wallis Test...")

        # log the results
        logger.info("------------------------------")
        logger.info("Kruskal-Wallis results for scholarly papers grouped by author centric framework")

        # call kruskal helper function to test author centric framework
        kruskal_auth_centric_framework = perform_test(articles, 'auth_label', numerical_features, feature_name_mapping, 'kruskal')
        print(kruskal_auth_centric_framework)
        logger.info("------------------------------")

        logger.info("Kruskal-Wallis results for scholarly papers grouped by external agent framework")

        # call kruskal helper function to test external agent centric framework
        kruskal_ext_agent_framework = perform_test(articles, 'extagent_label', numerical_features, feature_name_mapping, 'kruskal')
        print(kruskal_ext_agent_framework)
        logger.info("------------------------------")

if __name__ == '__main__':
    # create an argparse object
    parser = argparse.ArgumentParser(description='src/stats.py: Run statistical tests on Scholarly papers to determine significant observations on latent factors responsible for reproducibility.')

    # itemize all of the available command line arguments
    parser.add_argument('--badged_file_path', type=str, default='./data/papers_with_badged_information.csv',
                help='File path to the badged papers data CSV file.')
    parser.add_argument('--unbadged_file_path', type=str, default='./data/papers_without_badged_information.csv',
                help='File path to the unbadged papers data CSV file.')
    parser.add_argument('--run_test', type=str, choices=['anova', 'kruskal', 'shapiro', 'levene', 'all'], default='all',
                help='Specify which statistical test to run: ANOVA, Kruskal, Shapiro, Levene or all.')
    parser.add_argument('--feature_type', type=str, choices=['normal', 'scaled'], default='normal',
                help='Choose the type of features to use: Normal or Scaled.')

    # parse the arguments from the given choices
    args = parser.parse_args()

    # run the program
    run_statistical_analysis(args)