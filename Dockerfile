FROM asdggg/gscodec-studio:latest

ENV TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"
WORKDIR /workspace

# Install torch-scatter with sm_120 support
RUN pip install torch-scatter --no-build-isolation

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
