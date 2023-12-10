#!/bin/bash
set -e

# Install dependencies for building SQLite
sudo apt-get update
sudo apt-get install -y build-essential wget

# Define SQLite version
SQLITE_VERSION=3440200  # Replace with the version you need

# Download and extract SQLite source code
wget https://www.sqlite.org/2023/sqlite-autoconf-$SQLITE_VERSION.tar.gz
tar xvfz sqlite-autoconf-$SQLITE_VERSION.tar.gz
cd sqlite-autoconf-$SQLITE_VERSION

# Configure, compile, and install SQLite
./configure --prefix=/usr
make
sudo make install
