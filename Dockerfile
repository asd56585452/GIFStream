FROM asdggg/gscodec-studio:latest

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /workspace

# Install torch-scatter compiled from source with CUDA support for sm_120
RUN pip install torch-scatter --no-cache-dir --no-binary torch-scatter --no-build-isolation

# Install gridencoder
COPY third_party/gridencoder /workspace/third_party/gridencoder
RUN pip install --no-build-isolation -e /workspace/third_party/gridencoder

# Build and install MLEntropy with modern pybind11
COPY third_party/MLEntropy /workspace/third_party/MLEntropy
RUN cd /workspace/third_party/MLEntropy && \
    mkdir -p build && cd build && \
    cmake ../cpp -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Install GIFStream gsplat
COPY . /workspace
RUN pip install --no-build-isolation -e .

WORKDIR /workspace

