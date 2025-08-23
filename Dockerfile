FROM dealii/dealii:latest

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy as the 'dealii' user to avoid permission issues
COPY --chown=dealii:dealii . .

# Use the same non-root user the base image uses
USER dealii

# Clean any stray cache, then configure & build
RUN rm -rf build CMakeCache.txt CMakeFiles \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j

CMD ["./build/wave_solver"]
