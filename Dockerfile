FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#tag mean c9:CUDA Toolkit 9.0, d7:cuDNN7, tf15:Tensorflow R1.5.0
############ nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  already have below variables   ###########
# NVIDIA_REQUIRE_CUDA=cuda>=9.0
# CUDA_PKG_VERSION=9-0=9.0.176-1
# CUDA_VERSION=9.0.176
# CUDNN_VERSION=7.0.5.15
# NVIDIA_VISIBLE_DEVICES=all
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# NCCL_VERSION=2.1.4
# LIBRARY_PATH=/usr/local/cuda/lib64/stubs:


# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TF_VER=1.6.0rc0

#CUDA toolkit version
ENV CUDA_VER=9.0
ENV CUDNN_VER=7.0
ENV NCCL_VER=2.1.4

# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box
ENV PY_VER=3.5

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

############## install utilities #####################
RUN apt-get update && apt-get install -y --no-install-recommends  --allow-downgrades \
        build-essential \
        cmake \
        git \
        curl \
        vim \
	      pciutils \
        less \
	      nano \
        wget \
        ca-certificates \
        libnccl2=$NCCL_VER-1+cuda$CUDA_VER \
        libnccl-dev=$NCCL_VER-1+cuda$CUDA_VER \
        libjpeg-dev \
        libpng-dev \
        python$PY_VER \
        python$PY_VER-dev \
	      python`echo $PY_VER | cut -c1-1`-pip \
	      rsync \
	      swig \
	      unzip \
	      zip 	 

############ configure python ###################
ENV PYTHONIOENCODING utf-8
RUN rm -f  /usr/bin/python && \
    rm -f /usr/bin/python`echo $PY_VER | cut -c1-1` && \
    ln -s /usr/bin/python$PY_VER /usr/bin/python && \
    ln -s /usr/bin/python$PY_VER /usr/bin/python`echo $PY_VER | cut -c1-1`

RUN pip3 install pip --upgrade &&\
    pip3 install setuptools

######## install dependency files for tacotron  TODO  seperate for train/for preprocessing  #########
RUN pip3 install --no-cache-dir \
    appnope==0.1.0 \
    audioread==2.1.5 \
    beautifulsoup4==4.6.0 \
    bleach==1.5.0 \
    bs4==0.0.1 \
    cachetools==2.0.1 \
    certifi==2017.7.27.1 \
    chardet==3.0.4 \
    click==6.7 \
    cycler==0.10.0 \
    decorator==4.1.2 \
    dill==0.2.7.1 \
    ffprobe==0.5 \
    Flask==0.12.2 \
    Flask-Cors==3.0.3 \
    future==0.16.0 \
    gapic-google-cloud-datastore-v1==0.15.3 \
    gapic-google-cloud-error-reporting-v1beta1==0.15.3 \
    gapic-google-cloud-logging-v2==0.91.3 \
    gapic-google-cloud-pubsub-v1==0.15.4 \
    gapic-google-cloud-spanner-admin-database-v1==0.15.3 \
    gapic-google-cloud-spanner-admin-instance-v1==0.15.3 \
    gapic-google-cloud-spanner-v1==0.15.3 \
    google-auth==1.1.1 \
    google-cloud==0.27.0 \
    google-cloud-bigquery==0.26.0 \
    google-cloud-bigtable==0.26.0 \
    google-cloud-core==0.26.0 \
    google-cloud-datastore==1.2.0 \
    google-cloud-dns==0.26.0 \
    google-cloud-error-reporting==0.26.0 \
    google-cloud-language==0.27.0 \
    google-cloud-logging==1.2.0 \
    google-cloud-monitoring==0.26.0 \
    google-cloud-pubsub==0.27.0 \
    google-cloud-resource-manager==0.26.0 \
    google-cloud-runtimeconfig==0.26.0 \
    google-cloud-spanner==0.26.0 \
    google-cloud-speech==0.28.0 \
    google-cloud-storage==1.3.2 \
    google-cloud-translate==1.1.0 \
    google-cloud-videointelligence==0.25.0 \
    google-cloud-vision==0.26.0 \
    google-gax==0.15.15 \
    google-resumable-media==0.3.0 \
    googleapis-common-protos==1.5.3 \
    grpc-google-iam-v1==0.11.4 \
    grpcio==1.6.3 \
    html5lib==0.9999999 \
    httplib2==0.10.3 \
    idna==2.6 \
    ipdb==0.10.3 \
    ipython==6.2.1 \
    ipython-genutils==0.2.0 \
    iso8601==0.1.12 \
    itsdangerous==0.24 \
    jamo==0.4.1 \
    jedi==0.11.0 \
    Jinja2==2.9.6 \
    joblib==0.11 \
    librosa==0.5.1 \
    llvmlite==0.20.0 \
    m3u8==0.3.3 \
    Markdown==2.6.9 \
    MarkupSafe==1.0 \
    matplotlib==2.1.0 \
    monotonic==1.3 \
    nltk==3.2.5 \
    numba==0.35.0 \
    numpy==1.13.3 \
    oauth2client==3.0.0 \
    parso==0.1.0 \
    pexpect==4.2.1 \
    pickleshare==0.7.4 \
    ply==3.8 \
    prompt-toolkit==1.0.15 \
    proto-google-cloud-datastore-v1==0.90.4 \
    proto-google-cloud-error-reporting-v1beta1==0.15.3 \
    proto-google-cloud-logging-v2==0.91.3 \
    proto-google-cloud-pubsub-v1==0.15.4 \
    proto-google-cloud-spanner-admin-database-v1==0.15.3 \
    proto-google-cloud-spanner-admin-instance-v1==0.15.3 \
    proto-google-cloud-spanner-v1==0.15.3 \
    protobuf==3.4.0 \
    ptyprocess==0.5.2 \
    pyasn1==0.3.7 \
    pyasn1-modules==0.1.5 \
    pydub==0.20.0 \
    Pygments==2.2.0 \
    pyparsing==2.2.0 \
    python-dateutil==2.6.1 \
    pytz==2017.2 \
    requests==2.18.4 \
    resampy==0.2.0 \
    rsa==3.4.2 \
    scikit-learn==0.19.0 \
    scipy==0.19.1 \
    simplegeneric==0.8.1 \
    six==1.11.0 \
    tenacity==4.4.0 \
    #tensorflow-gpu==1.3.0 \
    #tensorflow-tensorboard==0.1.8 \
    tinytag==0.18.0 \
    tqdm==4.19.2 \
    traitlets==4.3.2 \
    urllib3==1.22 \
    wcwidth==0.1.7 \
    Werkzeug==0.12.2 \
    youtube-dl==2017.10.15.1 \
    unidecode==1.0.22 \
    inflect==0.2.5 

########## for https://github.com/keithito/tacotron 
RUN pip3 install --no-cache-dir \
    falcon==1.2.0  \
    inflect==0.2.5 \
    librosa==0.5.1 \
    matplotlib==2.0.2 \
    numpy==1.13.0 \
    scipy==0.19.0 \
    tqdm==4.11.2 \
    Unidecode==0.4.20 

    
#########  Install Tensorflow and Keras after OpenMPI Install  for debug ###############
RUN pip3 install --no-cache-dir tensorflow-gpu==$TF_VER

########   install some utilities #############
RUN pip3 install --no-cache-dir \
         h5py       \
         jupyter    \
         pandas    

WORKDIR "/examples"

######### port for TensorBoard and jupyter
EXPOSE 6006
EXPOSE 8888 

######## expose Library on NVIDIA-Docker environment
RUN export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH  &&\
    ldconfig $LD_LIBRARY_PATH

################ remove cache for apt-get ###################
RUN     rm -r /var/lib/apt/lists/*
