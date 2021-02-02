import agentMET4FOF_ml_extension.agentMET4FOF_ml_extension.agents as ml_agents
from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.agents import AgentNetwork
from agentMET4FOF_ml_extension.agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
import agent_demo.agents as pitchin_agents

def main():
    agentNetwork = AgentNetwork(dashboard_modules=[pitchin_agents, ml_agents],
                                dashboard_extensions=Dashboard_ML_Experiment,
                                backend="mesa")
if __name__ == "__main__":
    main()
