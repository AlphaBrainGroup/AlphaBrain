#!/usr/bin/env bash
# Wrapper for zhaoyiren's robocasa conda env python.
#
# Why: zhaoyiren's robocasa env has opencv-python (4.11.0.86) and mujoco that
# both need libGL.so.1 / libEGL.so.1 / libGLdispatch.so.0 at runtime — none of
# which exist on this machine's default LD path. We installed the missing
# render libraries (libGL, libEGL with proper eglQueryString export, libGLX,
# libGLdispatch, mesalib) via conda-forge into a self-contained shim env at
# /share/chenziyang/render_libs/, and expose them through LD_LIBRARY_PATH so
# zhaoyiren's robocasa python can find them.
#
# This is consistent with `feedback_no_modify_others_paths.md`: zhaoyiren's
# env is read-only; everything mutated lives under /share/chenziyang/.
#
# We also set the rendering backend env vars (MUJOCO_GL=egl,
# PYOPENGL_PLATFORM=egl) and disable numba JIT to match zhaoyiren's
# robocasa365 invocation pattern.

REAL_PYTHON=/share/zhaoyiren/conda_envs/robocasa/bin/python
RENDER_LIBS=/share/chenziyang/render_libs/lib

export LD_LIBRARY_PATH="${RENDER_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

exec "$REAL_PYTHON" "$@"
