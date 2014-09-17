name := "latent-pcfg"

version := "0.1.0 "

scalaVersion := "2.10.4"

retrieveManaged := true

javaOptions += "-Xmx4G"

libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-parser" % "2.0.4",
  "org.rogach" %% "scallop" % "0.9.5",
  "org.scalanlp" % "nak" % "1.2.1"
)
