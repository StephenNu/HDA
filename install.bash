#!/bin/bash
# remember to install weka to the classpath. In arch this can be done by.
# export CLASSPATH=/usr/share/java/weka/weka.jar:$CLASSPATH
function createJAR() {
  ant -f build_package.xml
  mkdir weka
  mkdir weka/filters
  mkdir weka/filters/supervised/
  mkdir weka/filters/supervised/attribute/
  cp java-build/classes/* weka/filters/supervised/attribute
  jar cf HDA.jar weka/filters weka/filters/supervised weka/filters/supervised/attribute/ weka/filters/supervised/attribute/*
  rm -R weka
}

function setup() {
  createJAR
  zip HDA.zip ./src/main/java/weka/filters/supervised/attribute/* ./HDA.jar ./build_package.xml ./Description.props
}

setup
