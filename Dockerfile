FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG USER_ID=1001
ARG GROUP_ID=101
ARG USER=app
ENV USER=$USER
ARG PROJ_ROOT=/$USER

USER root
RUN groupadd --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

RUN mkdir $PROJ_ROOT && chown $USER $PROJ_ROOT
WORKDIR $PROJ_ROOT	

# These next two folders will be where we will mount our local data and out
# directories. We create them manually (automatic creation when mounting will
# create them as being owned by root, and then our program cannot use them).
RUN mkdir data && chown $USER data
RUN mkdir out && chown $USER out

# tzdata configuration stops for an interactive prompt without the env var.
# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
# https://stackoverflow.com/questions/51023312/docker-having-issues-installing-apt-utils/56569081#56569081
ENV TZ=Europe/London
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
	apt-get install --no-install-recommends -y \
	tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	ca-certificates \
	git \
	libsm6 \
	libxext6 \
	libxrender-dev \
	jq \
	locales \
	libcairo2-dev \  
	libpango1.0-dev \
	pkg-config \	
	texlive \
	dvisvgm \
	dvipng \
	texlive-latex-extra \
	texlive-fonts-recommended \
	cm-super \
	pandoc \
	texlive-xetex \
	texlive-plain-generic \
	texlive-science \
	libmagickwand-dev \
	libxml2-dev \
	libxslt-dev \
	linux-tools-common \
	linux-tools-generic \
	ffmpeg && \
	rm -rf /var/lib/apt/lists/*

# Set the locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

###############################################################################
# Conda 
###############################################################################

ENV PATH=/home/$USER/mambaforge/bin:$PATH
RUN curl -sLo ./mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/22.9.0-1/Mambaforge-22.9.0-1-Linux-x86_64.sh \
 && chmod +x ./mambaforge.sh \
 && ./mambaforge.sh -b -p /home/$USER/mambaforge \
 && rm ./mambaforge.sh \
 && mamba clean -ya

###############################################################################
# /Conda
###############################################################################


###############################################################################
# Pip install
###############################################################################

RUN pip install --upgrade pip
COPY --chown=$USER pkg_requirements.txt pkg_requirements.txt
COPY --chown=$USER devel_requirements.txt devel_requirements.txt
COPY --chown=$USER private_requirements.txt private_requirements.txt
RUN pip install -r pkg_requirements.txt
RUN pip install -r devel_requirements.txt
RUN pip install -r private_requirements.txt

###############################################################################
# /Pip install
###############################################################################


###############################################################################
# JupyterLab 
###############################################################################

RUN conda install --yes -c conda-forge nodejs
# From: https://stackoverflow.com/questions/67050036/enable-jupyterlab-extensions-by-default-via-docker
COPY --chown=$USER configs/jupyter_notebook_config.py /etc/jupyter/
# Add the following file to allow extensions on startup (the file was created by 
# diffing a container running jupyterlab that had extensions manually enabled).
COPY --chown=$USER configs/plugin.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/
# Keyboard shortcuts
COPY --chown=$USER configs/shortcuts.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
# Getting some permission errors printed in terminal after running Jupyter Lab, 
# using the below line to fix it:
RUN chown -R $USER:$USER /home/$USER/.jupyter

###############################################################################
# /JupyterLab 
###############################################################################


# Temp install, to avoid cache miss.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#	<lib> \
#	<lib> && \
# 	rm -rf /var/lib/apt/lists/*
# RUN pip install <lib>

# In order to allow the Python package to be edited without a rebuild, install
# all code as a volume. We will still copy the files initially, so that things
# like the below pip install can work.
COPY --chown=$USER ./retinapy ./retinapy

# Install our own project as a module. This is done so the tests and JupyterLab
# code can import it.
RUN pip install -e ./retinapy
# To allow access to pybin
ENV PYTHONPATH=$PYTHONPATH:$PROJ_ROOT

# Switching to our new user. Do this at the end, as we need root permissions in
# order to create folders and install things.
USER $USER

