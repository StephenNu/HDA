#!/bin/bash
# remember to install weka to the classpath. In arch this can be done by.
# export CLASSPATH=/usr/share/java/weka/weka.jar:$CLASSPATH
function setup() {
	if [ $# = 0 ]; then
		ant -f build_package.xml -p
	else
		ant -f build_package.xml $1 -Dpackage=HDA
	fi
}

setup $*
