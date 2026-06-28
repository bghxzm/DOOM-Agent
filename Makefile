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
# Default target.  Ex: make or make ARGS="--agent=dt"
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

debug_agent:
	@make main ARGS="--agent=dt"

agent:
	@make main ARGS="--agent"

test_encoder:
	@make main ARGS="--enc"

test_environment:
	@make main ARGS="--env"
