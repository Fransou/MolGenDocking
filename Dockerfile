# ------------------------------------------------------------------------------------------------------------
FROM mambaorg/micromamba:latest AS base

USER root

RUN apt-get update \
    && apt-get install -y wget make g++ libboost-filesystem-dev libboost-system-dev \
       xutils-dev libxss1 xscreensaver xscreensaver-gl-extra xvfb \
    && rm -rf /var/lib/apt/lists/* /tmp/*


COPY docker_environment.yml pyproject.toml constraints_github_action.txt ./
COPY mol_gen_docking ./mol_gen_docking
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n base -f docker_environment.yml \
    && pip install . -c ./constraints_github_action.txt \
    && rm -rf /root/.cache/pip \
    && micromamba clean --all --yes

RUN wget -O /tmp/ADFRsuite.tar.gz https://ccsb.scripps.edu/adfr/download/1038/ \
    && tar -xzf /tmp/ADFRsuite.tar.gz -C /opt/ \
    && cd /opt/ADFRsuite_* \
    && echo "Y" | ./install.sh -d . -c 0 \
    && cd / \
    && rm -rf /tmp/ADFRsuite.tar.gz \
    && wget -O /tmp/vina.tgz --no-check-certificate https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz \
    && tar -xzf /tmp/vina.tgz -C /opt/ \
    && mv /opt/autodock_vina_1_1_2_linux_x86/bin/* /usr/local/bin/ \
    && rm -rf /tmp/vina.tgz /opt/autodock_vina_1_1_2_linux_x86
