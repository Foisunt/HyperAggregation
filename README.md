# HyperAggregation
This is the git repo for the IJCNN '24 paper HyperAggregation: Aggregating over Graph Edges with Hypernetworks

A link to the paper will be provided once available

The Code directory contains the code necessary to preprocess the data, train the models for all results reportet in the paper.
The Experiments directory contains one file pre number in the paper with the hyperparameters to train that model.

To reproduce results:
- install the requirements (we used Python 3.10.12, and see requiremenets.txt; major ones are: jupyterlab seaborn tqdm torch torchmetrics torch_geometric (+pyglib etc) ogb)
- preprocess the dataset (splits) using the PreprocessSplitSave notebook
- run the experiment you want, you can
  - do grid search by providing more than one hyperparameter in the experiment file, see search_GHC_RomanEmpire.yml as an example
  - reproduce the numbers of the paper by running the file with MODELNAME_SETTING_DATASET.yml, where SETTING is tr for transductive and in for inductive
  - to do so run "python runner.py EXPERIMENTNAME" or "python runner.py NGPU G1.G2....GN EXPERIMENTNAME" where NGPU is the number of GPUs that should be used and G1 etc are their IDs (e.g. 3 0.1.1 would run 3 processes, one on cuda:0 and two on cuda:1; might only work on linux, runner.py l. 37); EXPERIMENTNAME uses pathlib's glob to resolve wildcards, so *.yml would run all experiments (endig with .yml)
- the results notebook has code to plot the results of hyperparameter search and the ablation studies
- the reults-F notebook has code to get the test accuracy at the highest val acc of an experiment, show which hyperparameters reached that and print the test acc as latex

Note that the pyg loader for Zinc was iffy at the time of coding this (2023), we downloaded it manually and used the slightly modified zinc.py file for loading (see https://github.com/pyg-team/pytorch_geometric/issues/1602)
