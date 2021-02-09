# Multi Agent System and Bayesian Autoencoders for Quality Control in Manufacturing

This demonstrator is part of the Pitch-in project which demonstrates the use of distributed learning for quality control in manufacturing.
- We have developed a multi-agent system for a reconfigurable data pipeline and at the core of it, we employ the Bayesian Autoencoder, a novel deep learning algorithm which we have developed for quantifying uncertainty.
- We use the dataset from a real production line at Detroit, Michigan provided by Liveline Technologies. It is available at https://www.kaggle.com/supergus/multistage-continuousflow-manufacturing-process?select=notes_on_dataset.txt 
- To develop the demonstrator, we use the package `agentMET4FOF` (https://agentmet4fof.readthedocs.io/en/latest/)

## Running the code
- Place the csv dataset in the following path :  `/multi-stage-dataset/continuous-factory-process.csv`
- To run a blank demonstrator (without agents), execute : `agent_demo/run_main_blank.py`
- To run the full demonstrator (with agents pre-configured), execute : `agent_demo/run_main_full.py`
- Upon executing the code, you can access the demonstrator `http://localhost:8050/` via your browser

## Screenshots




## Acknowledgement 

This work is developed in PITCH-IN (Promoting the Internet of Things via Collaborations between HEIs and Industry) project funded by Research England.
