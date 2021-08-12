#!/bin/bash

function testing(){
    for file in `ls $1`
    do
      if [ "${file##*.}"x = "py"x ];then
        python $1/$file >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "Test[OK]" $1/$file
        else
            echo -e "Test[ERROR]" $1/$file
            exit 1
        fi
      fi
    done
}

testing tests/$1





