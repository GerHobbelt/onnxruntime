#!/bin/bash
set -e -x

# Development tools and libraries
yum -y install \
    graphviz

# Download a file from internet
function GetFile {
  local uri=$1
  local path=$2
  local force=${3:-false}
  local download_retries=${4:-5}
  local retry_wait_time_seconds=${5:-30}

  if [[ -f $path ]]; then
    if [[ $force = false ]]; then
      echo "File '$path' already exists. Skipping download"
      return 0
    else
      rm -rf $path
    fi
  fi

  if [[ -f $uri ]]; then
    echo "'$uri' is a file path, copying file to '$path'"
    cp $uri $path
    return $?
  fi

  echo "Downloading $uri"
  # Use aria2c if available, otherwise use curl
  if command -v aria2c > /dev/null; then
    aria2c -q -d $(dirname $path) -o $(basename $path) "$uri"
  else
    curl "$uri" -sSL --retry $download_retries --retry-delay $retry_wait_time_seconds --create-dirs -o "$path" --fail
  fi

  return $?
}

if [ ! -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/python/cp37-cp37m/bin/python3.7" "/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
GLIBC_VERSION=$(getconf GNU_LIBC_VERSION | cut -f 2 -d \.)

DISTRIBUTOR=$(lsb_release -i -s)

if [[ "$DISTRIBUTOR" = "CentOS" && $SYS_LONG_BIT = "64" ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi

cd /tmp/src

echo "Installing azcopy"
mkdir -p /tmp/azcopy
GetFile https://aka.ms/downloadazcopy-v10-linux /tmp/azcopy/azcopy.tar.gz
tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
cp /tmp/azcopy/azcopy /usr/bin

echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin
echo "Installing Node.js"
GetFile https://nodejs.org/dist/v16.14.2/node-v16.14.2-linux-x64.tar.gz /tmp/src/node-v16.14.2-linux-x64.tar.gz
tar --strip 1 -xf /tmp/src/node-v16.14.2-linux-x64.tar.gz -C /usr

cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps_aten\.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r ${0/%install_deps_aten\.sh/requirements\.txt}
  if ! [[ ${PYTHON_EXE} = "/opt/python/cp310-cp310/bin/python3.10" ]]; then
    ${PYTHON_EXE} -m pip install -r ${0/%install_deps_aten\.sh/..\/training\/ortmodule\/stage1\/requirements_torch_cpu\/requirements.txt}
  else
    ${PYTHON_EXE} -m pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
  fi
done

cd /tmp/src
GetFile 'https://sourceware.org/pub/valgrind/valgrind-3.16.1.tar.bz2' /tmp/src/valgrind-3.16.1.tar.bz2
tar -jxvf valgrind-3.16.1.tar.bz2
cd valgrind-3.16.1
./configure --prefix=/usr --libdir=/usr/lib64 --enable-only64bit --enable-tls
make -j$(getconf _NPROCESSORS_ONLN)
make install

cd /
rm -rf /tmp/src
