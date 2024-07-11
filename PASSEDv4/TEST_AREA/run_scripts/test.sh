#!/bin/bash

#+------------------------------------------------------------------------------+
#|                                   FUNCTION                                   |
#+------------------------------------------------------------------------------+
function PrintArray() {
        # Example: PrintArray ${array[@]}
        in=("$@")
        # Print out Level
        for item in ${in[*]}; do
                echo $item
        done
}

function SearchFiles() {
        # Example: files=$(SearchFiles $dir "dat")
        dir=$1
        suffix=$2
        files=($(find $dir -name "*.$suffix"))
        echo ${files[@]}
}

function ArrayUnique() {
        in=$1
        out=($(echo "${in[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

        echo ${out[@]}
}

function FindLevelSereis() {
        dir=$1
        # Search the level series
        files=$(SearchFiles $dir "dat")
        LEVEL=()
        for item in ${files[*]}; do
                # printf "   %s\n" $item
                # get basename
                name="$(basename -- $item)"
                # echo $name
                # string split "."
                str1=($(echo $name | tr '.' "\n"))
                # echo ${str1[1]}
                # string split "_"
                str2=($(echo ${str1[1]} | tr '_' "\n"))
                len=${#str2[@]}
                # printf "    len=%d\n" $len
                if [[ $len == 4 ]]; then
                        # printf "     %s\n" ${str2[3]}
                        # string split "l"
                        str3=($(echo ${str2[3]} | tr 'l' "\n"))
                        # printf "     %s\n" ${str3[1]}
                        LEVEL+=(${str3[1]})
                fi
        done

        echo ${LEVEL[@]}
}


# SCALE="0.1"
# SCALE="0.2"
# SCALE="0.3"
# SCALE="0.8"
# SCALE="1.0"
# SCALE="2.0"
# SCALE="3.0"
SCALE="5.0"
# SCALE="7.0"
# SCALE="10.0"
# SCALE="14.0"
# SCALE="16.0"
# SCALE="28.0"	# too long claculation time, the GPU will be escaped
# SCALE="20.0"
# SCALE="32.0"

SARMode="1"     # Stripmap
# SARMode="2"     # Spotlight


MODEL="../model/Di1_2m_0deg_17.363281/Di1_2m_0deg_17.363281.3ds"



PAR_CNF="../UD04_modified/UD04_SAR3_0.5.json"   # ThetaAz=2.0[deg]
# PAR_CNF="../UD04_modified/UD04_SAR3_1.0.json"   # ThetaAz=1.0[deg]
# PAR_CNF="../UD04_modified/UD04_SAR3_0.5.json"   # ThetaAz=0.5[deg]
# PAR_CNF="../UD04_modified/UD04_SAR3_0.3.json"   # ThetaAz=0.3[deg]
# PAR_CNF="../UD04_modified/UD04_SAR3_2.0_Asp045.json"    # ThetaAz=2.0[deg], Asp=45[deg]
# PAR_CNF="../UD04_modified/UD04_SAR3_2.0_Asp090.json"    # ThetaAz=2.0[deg], Asp=90[deg]

PAR_STV="../UD04_modified/UD04.stv"
MATERIAL="../UD04_modified/"
FOLDER_RCS="../res_Di3_AntPat/RCS/"
FOLDER_SAR="../res_Di3_AntPat/SAR/"
NAME="UD04MODIFIED"



PASSED_RCS="../../src/RCS_cuda/PASSEDv4_ORG2_TitanX"
PASSED_SAR="../../src/RDA/PASSEDv3_RDA_ORG2"


# echo "+------------------------+"
# echo "| 0. Remove folders      |"
# echo "+------------------------+"
rm -rf $FOLDER_RCS
rm -rf $FOLDER_SAR
mkdir -p $FOLDER_RCS
mkdir -p $FOLDER_SAR

echo "+----------------------+"
echo "| 1. RCS Simulation    |"
echo "+----------------------+"
cmd_RCS="$PASSED_RCS $PAR_CNF $PAR_STV $MATERIAL $MODEL $FOLDER_RCS $NAME -SC $SCALE -SARMODE $SARMode -GPU -PEC"
echo " cmd = $cmd_RCS"
echo ""
$cmd_RCS

echo "+----------------------+"
echo "| 2. SAR Simulation    |"
echo "+----------------------+"
cmd_SAR="$PASSED_SAR $PAR_CNF $PAR_STV $FOLDER_RCS $FOLDER_SAR $NAME"
echo " cmd = $cmd_SAR"
echo ""
$cmd_SAR

echo "+----------------------+"
echo "| 2-2. Each Level      |"
echo "+----------------------+"
# Find the level series
LevelSeries=$(FindLevelSereis $FOLDER_RCS)
Level=$(ArrayUnique "${LevelSeries[@]}")

# Process Each Level
for i in ${Level[*]}; do
        LV="-LV "$i
        cmd_SAR="$PASSED_SAR $PAR_CNF $PAR_STV $FOLDER_RCS $FOLDER_SAR $NAME $LV"
        echo " cmd = $cmd_SAR"
        echo ""
        $cmd_SAR
done




# echo "+-----------------------------+"
# echo "|  -TM 1. Internal dihedral   |"
# echo "+-----------------------------+"
# cmd_RCS="$PASSED_RCS $PAR_RCS $MODEL $MATERIAL $FOLDER_RCS $NAME -SC $SCALE -GPU -PEC -ST 1"
# # cmd_RCS="$PASSED_RCS $PAR_RCS $MODEL $MATERIAL $FOLDER_RCS $NAME -SC $SCALE -GPU -ST 1"
# echo " cmd = $cmd_RCS"
# echo ""
# $cmd_RCS

