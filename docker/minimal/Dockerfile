# syntax=docker/dockerfile:1.4
FROM ros:humble

# Language setup
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /server

# Install common packages
RUN apt-get update && apt upgrade -y
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
  python3-dev \
  python3-pip \
  curl \
  git \
  openssh-client \
  wget \
  vim \
  unzip \
  lsb-release \
  libboost-all-dev \
  libeigen3-dev \
  python3-numpy \
  python3-scipy \
  python3-dev \
  libassimp-dev \
  liburdfdom-dev \
  zsh

# Prepare SSH known_hosts to avoid interactive host key prompt for GitHub
RUN mkdir -p -m 0700 /root/.ssh \
  && ssh-keyscan -H github.com >> /root/.ssh/known_hosts

RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && \
  tar -C /opt -xzf nvim-linux-x86_64.tar.gz && \
  rm nvim-linux-x86_64.tar.gz

ENV PATH ${PATH}:/opt/nvim-linux-x86_64/bin

RUN git clone --depth=1 -b master https://github.com/perseusdg/config_files.git && \
  cd config_files && chmod +x install.sh && sh install.sh

RUN cd /server && git clone --recursive --depth=1 -b master https://github.com/stack-of-tasks/eigenpy.git && cd eigenpy && \
  mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j12 && make install

RUN cd /server && git clone --recursive --depth=1 -b master https://github.com/coal-library/coal.git && cd coal && mkdir build && \
  cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local  .. && make -j12 && make install

RUN cd /server && git clone --recursive --depth=1 -b master https://github.com/stack-of-tasks/pinocchio && cd pinocchio && mkdir build && \
  cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j12 && make install

# Add necessary environment variable to bashrc
RUN echo '\
  source /opt/ros/humble/setup.bash \n\
  if [ -d "/server/bunny_teleop_server" ] \n\
  then \n\
  source /server/bunny_teleop_server/ros/install/local_setup.bash \n\
  fi \n' >> $HOME/.zshrc



RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
  git+https://github.com/colcon/colcon-clean && \
  rm -r /root/.cache

RUN pip install pycollada

# Git clone and install
RUN cd /server && git clone https://github.com/demoPlz/bvp.git   bunny_teleop_server && chmod +x /server/bunny_teleop_server/bunny_teleop_server/nodes/bimanual_hand_detector_node.py


# Mount the SSH agent during pip install so git+ssh dependencies can be fetched
RUN --mount=type=ssh cd bunny_teleop_server \
  && GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" pip install . --no-deps  --use-deprecated legacy-resolver \
  && chmod +x /usr/local/lib/python3.10/dist-packages/bunny_teleop_server/nodes/bimanual_hand_detector_node.py



# Build ROS and chmod the file
RUN /bin/zsh -c ". /opt/ros/humble/setup.zsh && cd /server/bunny_teleop_server/ros && colcon clean workspace -y && colcon build"


