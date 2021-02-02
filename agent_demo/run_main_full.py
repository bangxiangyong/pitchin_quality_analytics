import agentMET4FOF_ml_extension.agentMET4FOF_ml_extension.agents as ml_agents
from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment

import agent_demo.agents as pitchin_agents
from util.plotting import plot_total_nll
import random

def main():
    first_nodes = 300
    second_nodes = 100
    likelihood = "homo_gaussian" # choose from {"1_gaussian", "homo_gaussian", "hetero_gaussian", "bernoulli", "cbernoulli"}
    simulate_batch_size = 50

    agentNetwork = AgentNetwork(dashboard_modules=[pitchin_agents],
                                dashboard_extensions=Dashboard_ML_Experiment,
                                backend="mesa")

    # instantiate agents
    machine_agent1 = agentNetwork.add_agent(name="DatastreamAgent-Stage1", agentType=pitchin_agents.Liveline_DatastreamAgent,
                                           input_stage=1,
                                           target_stage=1, simulate_batch_size=simulate_batch_size)
    bae_agent1 = agentNetwork.add_agent(name="BAE-Agent-1-"+likelihood, agentType=pitchin_agents.BAE_Agent, first_nodes=first_nodes,
                                        second_nodes= second_nodes, likelihood= likelihood)
    evaluate_agent1 = agentNetwork.add_agent(name="Evaluate-Agent-1", agentType=pitchin_agents.OOD_EvaluateAgent)
    postproc_agent1 = agentNetwork.add_agent(name="Plotting-Agent-1", agentType=pitchin_agents.GeneratePlotAgent, model_stage = 1)
    monitor_agent = agentNetwork.add_agent(name="MonitorAgent", agentType=MonitorAgent, custom_plot_function=plot_total_nll, buffer_size=1000000)

    # instantiate agents
    machine_agent2 = agentNetwork.add_agent(name="DatastreamAgent-Stage2", agentType=pitchin_agents.Liveline_DatastreamAgent,
                                           input_stage=2,
                                           target_stage=2, simulate_batch_size=simulate_batch_size)
    bae_agent2 = agentNetwork.add_agent(name="BAE-Agent-2-"+likelihood, agentType=pitchin_agents.BAE_Agent, first_nodes=first_nodes,
                                        second_nodes= second_nodes, likelihood= likelihood)
    evaluate_agent2 = agentNetwork.add_agent(name="Evaluate-Agent-2", agentType=pitchin_agents.OOD_EvaluateAgent)
    postproc_agent2 = agentNetwork.add_agent(name="Plotting-Agent-2", agentType=pitchin_agents.GeneratePlotAgent, model_stage = 2)

    # bind agents
    machine_agent1.bind_output(bae_agent1)
    bae_agent1.bind_output(evaluate_agent1)
    bae_agent1.bind_output(postproc_agent1)
    postproc_agent1.bind_output(monitor_agent)
    machine_agent2.bind_output(bae_agent2)
    bae_agent2.bind_output(evaluate_agent2)
    bae_agent2.bind_output(postproc_agent2)
    postproc_agent2.bind_output(monitor_agent)

    # add ML Experiment
    agentNetwork.add_ml_experiment(name="Process-Stage-1("+likelihood+")", agents=[machine_agent1,bae_agent1,evaluate_agent1], random_seed=random.randint(1,1000))
    agentNetwork.add_ml_experiment(name="Process-Stage-2("+likelihood+")", agents=[machine_agent2,bae_agent2,evaluate_agent2], random_seed=random.randint(1,1000))

    # set agent state to running
    agentNetwork.set_running_state()

if __name__ == "__main__":
    main()
