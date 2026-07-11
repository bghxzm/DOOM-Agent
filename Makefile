#
# Makefile to setup, train, and run the DOOM agent.
#
THIS_DIR=	$(shell pwd)
VENV=		${THIS_DIR}/venv
VBIN=		${VENV}/bin
PIP=		${VBIN}/pip3

DOOM_DIR=	${THIS_DIR}/doom_agent
TOOLS_DIR=	${DOOM_DIR}/tools
AGENT_DIR=	${DOOM_DIR}/agent
RUN_CMD=	${VBIN}/python3

REQS=		reqs.txt
ARGS=

.PHONY: main clean setup debug train agent


#
# Default target.  Ex: make or make ARGS="--agent=t"
#
main: ${VENV}
	@${RUN_CMD} ${DOOM_DIR}/$@.py ${ARGS}


#
# Clean repo to the default state.
#
clean:
	rm -rf venv


#
# Create the python virtual environment.
#
${VENV}:
	python3.11 -m venv venv
	${PIP} install -r ${REQS}

#
# Set up the virtual environment.
#
setup: ${VENV}


#
# Target agent options.
#
train_agent:
	@make main ARGS="--agent=t"

agent:
	@make main ARGS="--agent"

test_encoder:
	@make main ARGS="--encoder --test"

test_environment:
	@make main ARGS="--environment --test"

collect_data:
	@make main ARGS="--collect"

collect_data_episodes_3:
	@make main ARGS="--collect --episodes=3"

relabel_data:
	@make main ARGS="--relabel"

bc_train:
	@make main ARGS="--bc"

bc_train_epochs_5:
	@make main ARGS="--bc --epochs=5"

ppo_train:
	@make main ARGS="--ppo"

ppo_train_timesteps_2000:
	@make main ARGS="--ppo --timesteps=2000"
