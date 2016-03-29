#!/bin/sh
SHADOW_ROOT="/home/kklementiev/xop2.3/extensions/shadowvui/shadow-2.3.2m-linux"
PATH="${SHADOW_ROOT}/bin"
. $SHADOW_ROOT/.shadowrc.sh
echo 0 | trace -m menu
