cd /usr/local
mkdir java

cd /usr/local/java
wget 'https://github.com/frekele/oracle-java/releases/download/8u212-b10/jdk-8u212-linux-x64.tar.gz'

tar xzvf jdk-8u212-linux-x64.tar.gz && rm -r jdk-8u212-linux-x64.tar.gz

update-alternatives --install "/usr/bin/java" "java" "/usr/local/java/jdk1.8.0_212/bin/java" 1
update-alternatives --install "/usr/bin/javac" "javac" "/usr/local/java/jdk1.8.0_212/bin/javac" 1
# update-alternatives --install "/usr/bin/jshell" "jshell" "/usr/local/java/jdk1.8.0_212/bin/jshell" 1

update-alternatives --set java "/usr/local/java/jdk1.8.0_212/bin/java"
update-alternatives --set javac "/usr/local/java/jdk1.8.0_212/bin/javac"
# update-alternatives --set jshell "/usr/local/java/jdk1.8.0_212/bin/jshell"
