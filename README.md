# HDA
Heteroscedastic Discriminant Analysis (HDA) is a machine learning algorithm for dimensionality reduction. It is implemented as a package for the open source program WEKA.

#Building
Ensure you have the following programs and tools installed on your computer before proceeding.
* [Java's JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
* [apache-ant](http://ant.apache.org/)
* [Weka](http://www.cs.waikato.ac.nz/ml/weka/)
* [junit](http://junit.org/)

The next step is to ensure that weka.jar is in the $CLASSPATH variable.
On Arch Linux if weka was installed from the AUR, the jar is found in /usr/share/java/weka/weka.jar.
To add it to the CLASSPATH on Arch add this line to your ~/.bashrc
```
export CLASSPATH=/usr/share/java/weka/weka.jar:$CLASSPATH
```

Next step is clone the repo,
```
git clone https://github.com/StephenNu/HDA.git
```

After that move into the location of the repo and make the package by the following commands
```
cd HDA
./build.bash make_package
```

Now in weka open the package manager and click "File/URL" under the unoffical section.
Navigate to where the repo was cloned to, and go into the **dist/** folder, and select HDA.zip
Click okay, restart weka and HDA will be installed under filters.

#Tests
Once HDA is installed, you can run the HDA tests by going to the weka command line, and running
the following command
```
java weka.filters.HDATest
```
