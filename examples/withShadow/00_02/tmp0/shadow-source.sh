#!/bin/bash
SHADOW_ROOT="/home/klmnKubuntu/xop2.3/extensions/shadowvui/shadow-2.3.2m-linux"
PATH="${SHADOW_ROOT}/bin"
. $SHADOW_ROOT/.shadowrc.sh
epath < xsh_epath_tmp.inp
nphoton < xsh_nphoton_tmp.inp
input_source < xsh_input_source_tmp.inp
gen_source start.00