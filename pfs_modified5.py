import os
import sys, getopt
import numpy as np
from scipy.stats import rankdata
import networkx as nx
from time import time
from numpy.random import choice
from math import sqrt
import pandas as pd

P_VAL_CUT_OFF = 0.4


# returns gene expression matrix
def load_file(fi_name):
    labels, data = [], []
    fi = open(fi_name, 'r')
    header = fi.readline()
    skipped_gcount = 0

    # file --> 2-d array (genes x samples)
    # skip lines (genes) with missing values
    for line in fi:
        ln = line.strip().split('\t')
        row = ln[1:]
        try:
            row = list(map(float, ln[1:]))
            data.append(row)
            labels.append(ln[0])
        except:
            skipped_gcount += 1

    return np.array(data), labels, skipped_gcount


# returns a graph per pathway
def load_pathways(fpw_name):
    pw2G = {}

    # populate dict pw2G: key (pathway name), value (gene graph)
    fpw = open(fpw_name, 'r')
    header = fpw.readline()
    for line in fpw:
        row = line.strip().split('\t')

        if len(row) == 4:
            # Remove the index value
            row = row[1:]

        if len(row) >= 3:
            pw, node_1, node_2 = row
            if pw not in pw2G:
                pw2G[pw] = nx.Graph()
            pw2G[pw].add_edge(node_1, node_2)
    fpw.close()

    return pw2G


# filters GE matrix to contain expression of common genes in identical order
def rearrange_gdata(Xc, Xt, Xc_genes, Xt_genes):
    Xc_genes, Xt_genes = list(Xc_genes), list(Xt_genes)
    X_genes = list(set(Xc_genes) & set(Xt_genes))
    genes_ci, genes_ti = [], []

    # get indices for genes common to both groups
    for gene in X_genes:
        genes_ci.append(Xc_genes.index(gene))
        genes_ti.append(Xt_genes.index(gene))

    # filter expression matrix to include common genes only
    Xc = Xc[genes_ci]
    Xt = Xt[genes_ti]

    mapping = dict(zip(X_genes, range(len(X_genes))))

    return Xc, Xt, X_genes, mapping


# Compute the two distributions for one person
# Xc need to be a dictionnary "person, gene" -> expression
def subnetwork_distributions(target, target_gene_expressions, subnetwork, Xc, Xt, mapping):
    """
    Compute the two distributions for one person on one subnetwork
    target: tuple (index, relapse or non-relapse)
    """
    delta_t = []
    delta_c = []

    genes = subnetwork.nodes()

    # For each gene in the subnetwork
    for gene in genes:
        index_gene = mapping[gene]
        target_gene_expression = target_gene_expressions[index_gene]
        # For each person, compute the difference between gene expressions
        for index_person in range(np.shape(Xc)[1]):
            if (target[1] == 1 or target[0] != index_person):
                delta_c.append(target_gene_expression - Xc[index_gene][index_person])
        for index_person in range(np.shape(Xt)[1]):
            if (target[1] == 0 or target[0] != index_person):
                delta_t.append(target_gene_expression - Xt[index_gene][index_person])

    return delta_c, delta_t


# computes weights of genes in patients
def compute_weights(X, genes, theta_1, theta_2):
    num_genes, num_samples = np.shape(X)

    # function calculating vote to a gene from a patient, given quantiles in the group
    def get_vote(r, q1, q2):
        if r >= q1:
            return 1
        if r >= q2:
            return float(r - q2) / (q1 - q2)
        else:
            return 0

    # for each sample, calculate gene expression ranks: output matrix -- samples x genes
    ranks = [rankdata(X[:, j], 'dense') / float(num_genes) for j in range(num_samples)]

    # for each sample, calculate theta_1 and theta_2 quantiles for gene expression
    q1 = np.percentile(ranks, theta_1 * 100, axis=1)
    q2 = np.percentile(ranks, theta_2 * 100, axis=1)

    # 'weights' is a dict; gene (key) --> fuzzy vote in all samples (value: list)
    weights = {genes[i]: [get_vote(ranks[j][i], q1[j], q2[j]) for j in range(num_samples)] for i in range(num_genes)}

    return weights


# generates subnets using highly expressed genes
def generate_subnets(pw2G, he_genes):
    subnets = {}

    # use genes to induce subgraphs on pathways
    for pw in pw2G:
        s_count = 0
        pw_subgraph = pw2G[pw].subgraph(he_genes)
        for S in nx.connected_component_subgraphs(pw_subgraph):
            if S.number_of_nodes() >= 5:
                subnets[pw + '_' + str(s_count)] = S
                s_count += 1

    return subnets


def select_person(index, X):
    expressions = []
    for row in X:
        expressions.append(row[index])
    return expressions


def get_features_from_distribution(distribution_c, distribution_t, quantiles=[0.5, 0.25, 0.75]):
    features_c = np.quantile(distribution_c, quantiles)
    features_t = np.quantile(distribution_t, quantiles)
    return features_c, features_t


# pfsnet: generate subnetworks, calculate scores, perform permutation test
def pfsnet(Xc, Xt, X_genes, pw2G, beta, theta_1, theta_2, n_permutations):
    '''subnetwork generation'''

    # generating subnets, computing scores
    subnets_c, subnets_t, subnet_scores_c, subnet_scores_t = pfsnet_core(Xc, Xt, X_genes, pw2G, beta, theta_1, theta_2)

    '''permutation test test'''

    print('performing permutation test...')
    num_c, num_t = np.shape(Xc)[1], np.shape(Xt)[1]
    seq_c, seq_t = set(range(num_c)), set(range(num_t))
    subnet_scores_null_c, subnet_scores_null_t = {}, {}
    for n in range(n_permutations):
        # '''
        if (n + 1) % 100 == 0:
            print('iter:'), (n + 1)
        # '''
        # 'choose' samples (sampling without replacement)
        s_choice = choice(num_c + num_t - 1, size=num_c, replace=False)
        from_c_to_c = set([s for s in s_choice if s < num_c])
        from_t_to_c = set([s - num_c for s in s_choice if s >= num_c])
        from_c_to_t = seq_c - from_c_to_c
        from_t_to_t = seq_t - from_t_to_c
        Xc_new = np.c_[Xc[:, list(from_c_to_c)], Xt[:, list(from_t_to_c)]]
        Xt_new = np.c_[Xc[:, list(from_c_to_t)], Xt[:, list(from_t_to_t)]]
        subnet_scores_null_c[n], subnet_scores_null_t[n] = pfsnet_iter(Xc_new, Xt_new, X_genes, subnets_c, subnets_t,
                                                                       theta_1, theta_2)

    return subnets_c, subnets_t, subnet_scores_c, subnet_scores_t, subnet_scores_null_c, subnet_scores_null_t


# this function generates subnetworks and calculates their scores over the actual dataset
def pfsnet_core(Xc, Xt, X_genes, pw2G, beta, theta_1, theta_2):
    """computing weights"""

    print('generating subnetworks...')

    # get gene weights in 2 patient groups
    weights_c = compute_weights(Xc, X_genes, theta_1, theta_2)
    weights_t = compute_weights(Xt, X_genes, theta_1, theta_2)

    # find genes whose mean vote is greater than beta
    he_genes_c = [g for (g, w) in weights_c.items() if np.mean(w) >= beta]
    he_genes_t = [g for (g, w) in weights_t.items() if np.mean(w) >= beta]

    """subnetwork generation"""

    # extract subnets
    subnets_c = generate_subnets(pw2G, he_genes_c)
    subnets_t = generate_subnets(pw2G, he_genes_t)

    """subnetwork scoring"""

    print('scoring subnetworks...')

    # for each sample, get subnet scores
    subnet_scores_c, scores_1_c, scores_2_c = get_subnet_scores(subnets_c, weights_c, weights_t, np.shape(Xc)[1])
    subnet_scores_t, scores_1_t, scores_2_t = get_subnet_scores(subnets_t, weights_t, weights_c, np.shape(Xt)[1])

    return subnets_c, subnets_t, subnet_scores_c, subnet_scores_t


# this function calculates subnet scores over randomized datasets
def pfsnet_iter(Xc, Xt, X_genes, subnets_c, subnets_t, theta_1, theta_2):
    """computing weights"""

    # get gene weights in 2 phenotypes
    weights_c = compute_weights(Xc, X_genes, theta_1, theta_2)
    weights_t = compute_weights(Xt, X_genes, theta_1, theta_2)

    """subnetwork scoring"""

    # for each sample, get subnet scores
    subnet_scores_c, scores_1_c, scores_2_c = get_subnet_scores(subnets_c, weights_c, weights_t, np.shape(Xc)[1])
    subnet_scores_t, scores_1_t, scores_2_t = get_subnet_scores(subnets_t, weights_t, weights_c, np.shape(Xt)[1])

    return subnet_scores_c, subnet_scores_t


# this function returns significant subnets given subnet scores & null dist.
def get_sgnf_subnets(subnet_scores, subnet_scores_null):
    # dict: contains significant subnets (key) with their p-values (values)
    sgnf_subnets = {}

    for subnet in subnet_scores:
        null_dist = [subnet_scores_null[n][subnet] for n in subnet_scores_null]
        # p-value is calculated as proportion of points in null dist. with a greater score
        if subnet_scores[subnet] > 0:
            p_val = np.mean([(point) > (subnet_scores[subnet]) for point in null_dist])
            if p_val <= P_VAL_CUT_OFF:
                sgnf_subnets[subnet] = p_val

    return sgnf_subnets


# this function computes scores for each patient-subnetwork pair in a given class
def get_subnet_scores(subnets, weights_1, weights_2, num_samples):
    # scores is a dictionary which stores the scores (values) of all patients for each subnet (key)
    scores_1 = {s_name: {} for s_name in subnets}
    scores_2 = {s_name: {} for s_name in subnets}
    subnet_scores = {}

    # for each patient:
    # score 1: sum over --> (avg fuzzy vote of gene in type 1) x (fuzzy vote to gene from patient)
    # score 2: sum over --> (avg fuzzy vote of gene in type 2) x (fuzzy vote to gene from patient)

    for s_name in subnets:
        S_nodes = subnets[s_name].nodes()
        mean_weights_1 = {g: np.mean(weights_1[g]) for g in S_nodes}
        mean_weights_2 = {g: np.mean(weights_2[g]) for g in S_nodes}
        for j in range(num_samples):
            scores_1[s_name][j] = sum(mean_weights_1[g] * weights_1[g][j] for g in S_nodes)
            scores_2[s_name][j] = sum(mean_weights_2[g] * weights_1[g][j] for g in S_nodes)

    # to add to t-statistic denominator when variance of difference is zero
    epsilon = 10 ** (-10)

    # for each sample, get per subnet t-statistic value
    for s_name in subnets:
        s1 = np.array(list(scores_1[s_name].values()))
        s2 = np.array(list(scores_2[s_name].values()))

        # paired t-test
        x = s1 - s2
        mean_x = np.mean(x)
        std_err_x = sqrt(np.var(x) / len(x))

        subnet_scores[s_name] = mean_x / (std_err_x + epsilon)

    return subnet_scores, scores_1, scores_2


def write_scores_to_csv(features, name):
    """
    Put in pandas Dataframe as a one column Dataframe finishing by the label
    """
    dataframe = pd.DataFrame(features)
    dataframe.to_csv(name, sep='\t', index=False)
    return


def generate_scores(patient, subnets, Xc, Xt, mapping):
    name = "patient_" + str(patient[0])

    if (patient[1] == 0):
        target_gene_expressions = select_person(patient[0], Xc)
        name += "_Xc"
    else:
        name += "_Xt"
        target_gene_expressions = select_person(patient[0], Xt)

    print("Computing scores for patient: " + name)

    features = np.array([])

    for subnet in subnets.keys():
        print(" --- Subnet: " + str(subnet))
        distribution_c, distribution_t = subnetwork_distributions(patient, target_gene_expressions, subnets[subnet], Xc,
                                                                  Xt, mapping)
        scores_c, scores_t = get_features_from_distribution(distribution_c, distribution_t)
        features = np.concatenate((features, scores_c))
        features = np.concatenate((features, scores_t))
    features = np.concatenate((features, [patient[1]]))

    print(" --- writing to .csv")
    return features


def generate_every_scores(subnets, Xc, Xt, mapping):
    feature_matrix = []

    print("Generating control scores")
    for i in range(np.shape(Xc)[1]):
        features = generate_scores((i, 0), subnets, Xc, Xt, mapping)
        feature_matrix.append(features)

    print("Generating test scores")
    for i in range(np.shape(Xt)[1]):
        features = generate_scores((i, 1), subnets, Xc, Xt, mapping)
        feature_matrix.append(features)

    return pd.DataFrame(feature_matrix).T


def select_significant_networks(subnets_c, subnets_t, sgnf_subnets_c, sgnf_subnets_t):
    significant_networks = {}

    for sign_subnetwork in sgnf_subnets_c.keys():
        significant_networks[sign_subnetwork] = subnets_c[sign_subnetwork]

    for sign_subnetwork in sgnf_subnets_t.keys():
        significant_networks[sign_subnetwork] = subnets_t[sign_subnetwork]

    return significant_networks


def main(fei_c_name, fei_t_name, fpw_name, beta=0.5, theta_1=0.95, theta_2=0.85, n_permutations=1000):
    start_time = time()

    '''argument handling'''

    # fei_c_name, fei_t_name, fpw_name, beta, theta_1, theta_2, n_permutations = handle_args(argv)

    '''loading files'''

    print('loading input files...')

    # control/normal group
    Xc, Xc_genes, skipped_c = load_file(fei_c_name)
    print(Xc)
    if (skipped_c):
        print('Control group: missing data -- skipped ' + str(skipped_c) + ' lines')

    # test/disease group
    Xt, Xt_genes, skipped_t = load_file(fei_t_name)
    if (skipped_t):
        print('Test group: missing data -- skipped ' + str(skipped_t) + ' lines')

    # pathways
    pw2G = load_pathways(fpw_name)

    # removing genes not common to test and control
    Xc, Xt, X_genes, mapping = rearrange_gdata(Xc, Xt, Xc_genes, Xt_genes)

    # subnets_c, subnets_t, subnet_scores_c, subnet_scores_t = pfsnet_core(Xc, Xt, X_genes, pw2G, beta, theta_1, theta_2)

    # Run original PFSNet to select the most differencial pathways.
    subnets_c, subnets_t, subnet_scores_c, subnet_scores_t, subnet_scores_null_c, subnet_scores_null_t = pfsnet(Xc, Xt,
                                                                                                                X_genes,
                                                                                                                pw2G,
                                                                                                                beta,
                                                                                                                theta_1,
                                                                                                                theta_2,
                                                                                                                n_permutations)

    sgnf_subnets_c = get_sgnf_subnets(subnet_scores_c, subnet_scores_null_c)
    sgnf_subnets_t = get_sgnf_subnets(subnet_scores_t, subnet_scores_null_t)

    significant_networks = select_significant_networks(subnets_c, subnets_t, sgnf_subnets_c, sgnf_subnets_t)

    print("significant Networks")
    print(significant_networks)

    feature_matrix = generate_every_scores(significant_networks, Xc, Xt, mapping)

    feature_matrix.to_csv("feature_matrix", sep='\t')

    return feature_matrix


main("clean3_pfsnet_non_relapse_cleanedAgain.txt", "clean3_pfsnet_relapse_cleanedAgain.txt", "pfsnet_pathways_cleanedAgain2.txt")
#main("clean_pfsnet_nonrelapse.txt", "clean_pfsnet_relapse.txt", "pfsnet_pathways_cleaned2.txt")
