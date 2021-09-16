FROM darribas/gds_dev:6.1

# so I have write permissions for mounted volume
USER root
ENV NB_USER=jovyan
ENV NB_UID=1001
ENV NB_GID=1001

# install my favourite extra packages for jupyter
RUN jupyter labextension install @axlair/jupyterlab_vim \
     && jupyter labextension enable @axlair/jupyterlab_vim

RUN pip install git+https://github.com/peterprescott/nbhd

# run jupyter lab
CMD jupyter lab --no-browser --allow-root --NotebookApp.token='secret'
