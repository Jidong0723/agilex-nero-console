# Third-party notices

This repository contains local source snapshots so the hardware-validated NERO v120 installation remains reproducible.

## pyAgxArm

- Upstream: <https://github.com/agilexrobotics/pyAgxArm>
- Location: `vendor/pyAgxArm`
- Copyright: AgileX Robotics Co., Ltd. and contributors
- License: GNU Lesser General Public License v3.0 only (`LGPL-3.0-only`)
- License text: `vendor/pyAgxArm/LICENSE`

The NERO Control Console is an application that uses this library. The library remains separately installable and replaceable under the conditions of the LGPL.

## python-can-agx-cando

- Upstream: <https://github.com/agilexrobotics/python-can-agx-cando>
- Location: `vendor/python-can-agx-cando`
- Copyright: upstream authors and contributors
- License: MIT

The bundled snapshot contains the CANDO backend and upstream-distributed native DLLs required on Windows. Refer to the upstream repository for current hardware support and notices.

## NERO robot description

- Location: `vendor/nero_description`
- Origin: NERO/AgileX robot-description material used for kinematics

The robot description is provided for interoperability with NERO hardware. Product names and trademarks belong to their respective owners. This is an independent community project, not an official AgileX product.
