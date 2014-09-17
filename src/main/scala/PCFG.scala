package latent_pcfg

import java.io._
import edu.stanford.nlp._
import edu.stanford.nlp.trees._
import collection.JavaConversions._
import collection.JavaConverters._
import breeze.stats.distributions._
import breeze.linalg._
import edu.stanford.nlp.ling.StringLabel
import edu.stanford.nlp.tagger.maxent.MaxentTagger
import nak.NakContext._
import nak.core._
import nak.data._
import nak.cluster._
import nak.liblinear.LiblinearConfig
import breeze.linalg.SparseVector
import nak.util.ConfusionMatrix
import Util._

object pcfg {
	
	def main(args: Array[String]): Unit = {
		val opts = LatentPCFGOpts(args)

		if (!opts.unsup()) {  // Supervised Training
			val (trees,evalTrees) = read_trees(opts.dir(),opts.language(),opts.maxSentLength())
			val trainingTrees     = new Treebank(trees)
			val evaluationTrees   = new Treebank(evalTrees)
			val goldEvalTrees     = evalTrees.map(bt => bt.tree.deepCopy)

			// UNK insertion for training - low frequency words converted to UNK and clustered.
			println("Doing UNK Clustering.")
			val unkClassifier = loadClassifier[FeaturizedClassifier[String,String]](opts.unkModel())
			val rawWords = trainingTrees.getTrees.toList.map(t => t.tree.yieldWords.map(w => w.value).toList).flatMap(identity)
			val lowFreq = rawWords.groupBy(identity).mapValues(_.size).filter(_._2 < opts.unkCutoff()).map(_._1)

			trainingTrees.getTrees.foreach{ bt =>
				val forms = List("#","#","#") ++ bt.tree.yieldWords.map(w => w.value).toList ++ List("#","#","#")
        		val pos = List("#","#","#") ++ bt.tree.preOrderNodeList.filter(_.isPreTerminal).map(_.label.value).toList ++ List("#","#","#")

        		var idx = 0
        		forms.foreach{ form =>
        			if (lowFreq.contains(form)) {
        				//Extract Features
        				val features = forms(idx-3)+" "+forms(idx-2)+" "+forms(idx-1)+" "+forms(idx+1)+" "+forms(idx+2)+" "+forms(idx+3)+" "+
        				pos(idx-3)+" "+pos(idx-2)+" "+pos(idx-1)+" "+pos(idx)+" "+pos(idx+1)+" "+pos(idx+2)+" "+pos(idx+3)
        				
        				//Classify & Replace Label
        				bt.tree.preOrderNodeList.filter(_.isLeaf).apply(idx-3).setLabel(new StringLabel("UNK"+unkClassifier.predict(features)+"_T"))

        				
        			}
        			idx += 1
        		}
        		bt.tree.pennPrint
			}

			// UNK in eval
			println("")
			val evalUnks = evaluationTrees.getTrees.toList.map(t => t.tree.yieldWords.map(w => w.value).toList)
																							.flatMap(identity).toSet
																							.filter(!rawWords.contains(_))
			evaluationTrees.getTrees.foreach{ bt =>
				val forms = List("#","#","#") ++ bt.tree.yieldWords.map(w => w.value).toList ++ List("#","#","#")
        		val pos = List("#","#","#") ++ bt.tree.preOrderNodeList.filter(_.isPreTerminal).map(_.label.value).toList ++ List("#","#","#")

        		var idx = 0
        		forms.foreach{ form =>
        			if (evalUnks.contains(form)) {
        				//Extract Features
        				val features = forms(idx-3)+" "+forms(idx-2)+" "+forms(idx-1)+" "+forms(idx+1)+" "+forms(idx+2)+" "+forms(idx+3)+" "+
        				pos(idx-3)+" "+pos(idx-2)+" "+pos(idx-1)+" "+pos(idx)+" "+pos(idx+1)+" "+pos(idx+2)+" "+pos(idx+3)
        				
        				//Classify & Replace Label
        				bt.tree.preOrderNodeList.filter(_.isLeaf).apply(idx-3).setLabel(new StringLabel("UNK"+unkClassifier.predict(features)+"_T"))

        				
        			}
        			idx += 1
        		}
        		bt.tree.pennPrint
			}

			val target =  if (opts.language() == "zh") "IP" else "S"
			val grammar = new SeededGrammar(trainingTrees,evaluationTrees,target)


			println("Grammar Initialized, updating Theta.")
			grammar.seedAnnotations(opts.annotations(),false)
			grammar.annotateTrees
			grammar.updateAnnotatedTheta("train")
			grammar.updateTheta("train")

			println("Beginning Training.")
			for( iteration <- 1 to opts.iterationCount()) {
				val timeStart = System.currentTimeMillis
			
				grammar.resampleTrees
				if (opts.annotations() > 0) grammar.updateAnnotatedTheta("eval") else grammar.updateTheta("eval")
				val out = new File("eval-guess-"+iteration+".mrg")
				val pw = new PrintWriter(out)
				grammar.printOutTrees("eval",pw)
				pw.close

				println("Iteration #"+iteration+" Complete. Took "+(System.currentTimeMillis - timeStart)/1000.0+"s")
			}

			// Output Final Trees
			val outGoldFile  = new File("eval-gold.mrg")
			val pwgold  = new PrintWriter(outGoldFile)
			
			goldEvalTrees.foreach(tree => tree.pennPrint(pwgold))
	   		
	   		// Read in full Eval Tree Progression
			var evalTreeProgression = List[List[Tree]]()
			for( iteration <- 1 to opts.iterationCount()) {
				val fis = new FileInputStream("eval-guess-"+iteration+".mrg")
	        	val in = new InputStreamReader(fis, "UTF-8");
	        	val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
	        	val tr = new PennTreeReader(in)
	        	var evalTrees = List[Tree]()
	        	var curTree = tr.readTree
	        	while (curTree != null) {

	            	evalTrees = evalTrees ++ List(curTree)
	            	curTree = tr.readTree
	       		}
	       		evalTreeProgression ::= evalTrees

	       		val evalIter = new File("eval-guess-"+iteration+".mrg")
				evalIter.delete()
			}

			// Select mode-tree for each eval sentence.
			val out = new File("eval-guess.mrg")
			val pw = new PrintWriter(out)
			for (treeIndex <- 0 to evalTreeProgression(0).size-1) {
				val mode = evalTreeProgression.map(treeList => treeList(treeIndex)).groupBy(identity).maxBy(_._2.size)._1
				mode.pennPrint(pw)
			}
			pw.close
			pwgold.close
			
		} else { // Unsupervised
			println("Loading Tagger: "+opts.tagger())
			val tagger = new MaxentTagger(opts.tagger())
			val (trainSents,evalSents) = read_sentences(opts.dir(),opts.maxSentLength())

			println("\tLoaded "+trainSents.size+" training sentences.")			
			println("\tLoaded "+evalSents.size+" eval sentences.")

			// Either Tag or Construct Gold Tag Sequences
			var tagTrainSents = Array[String]()
			var tagEvalSents = Array[String]()
			if (!opts.goldTags()) {
				tagTrainSents = trainSents.map{ ts =>
					val tsTagged = tagger.tagTokenizedString(ts)
					tsTagged.diff(ts).split("_").drop(1).mkString(" ")
				}
				tagEvalSents = evalSents.map{ ts =>
					val tsTagged = tagger.tagTokenizedString(ts)
					tsTagged.diff(ts).split("_").drop(1).mkString(" ")
				}
			} else {
				tagTrainSents = trainSents
				tagEvalSents = evalSents
			}

			val grammar = new UnsupervisedGrammar(tagTrainSents,tagEvalSents,trainSents,evalSents,opts.pLex())
			grammar.seedTheta

			// Small treeset jumpstart
			if (opts.jumpTrees() != "") {
				val fis = new FileInputStream(opts.jumpTrees())
		        val in = new InputStreamReader(fis, "UTF-8");
		        val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
		        val tr = new PennTreeReader(in)
		        var trees = List[Tree]()

		        var curTree = tr.readTree
		        while (curTree != null) {

		            trees = trees ++ List(curTree)
		            curTree = tr.readTree
		        }
				val bank = new Treebank(trees.toArray.map(x => new BasicTree(x)))
				grammar.jumpStartTheta(bank)
			}

			var badTrees = List[Int]()

			for( iteration <- 1 to opts.iterationCount()) {
				println("Sampling Iteration: "+iteration)
				badTrees = grammar.resampleTrees
				if (opts.sm() > 0.0) grammar.calculateSmoothing
				grammar.updateTheta("train")

				val out = new File("eval-guess-"+iteration+".mrg")
				val pw = new PrintWriter(out)
				grammar.printOutTrees("eval",pw)
				pw.close
			}

			// Load Gold Eval
			val fis = new FileInputStream(opts.dir()+"/testing/dev-bin.mrg")
	        val in = new InputStreamReader(fis, "UTF-8");
	        val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
	        val tr = new PennTreeReader(in)
	        var goldEvaltrees = List[Tree]()
	        var curTree = tr.readTree
	        while (curTree != null) {

	            goldEvaltrees = goldEvaltrees ++ List(curTree)
	            curTree = tr.readTree
	        }
	        var out = new File("eval-gold.mrg")
			var pw = new PrintWriter(out)
	        goldEvaltrees.foreach{ tree =>
	        	if (tree.yieldWords.size/2 <= opts.maxSentLength()) {
	        		tree.pennPrint(pw)
	        	}
	        }
	        pw.close

	        // Read in full Eval Tree Progression
			var evalTreeProgression = List[List[Tree]]()
			for( iteration <- 1 to opts.iterationCount()) {
				val fis = new FileInputStream("eval-guess-"+iteration+".mrg")
	        	val in = new InputStreamReader(fis, "UTF-8");
	        	val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
	        	val tr = new PennTreeReader(in)
	        	var evalTrees = List[Tree]()
	        	var curTree = tr.readTree
	        	while (curTree != null) {

	            	evalTrees = evalTrees ++ List(curTree)
	            	curTree = tr.readTree
	       		}
	       		evalTreeProgression ::= evalTrees

	       		val evalIter = new File("eval-guess-"+iteration+".mrg")
				evalIter.delete()
			}

			// Select mode-tree for each eval sentence.
			out = new File("eval-guess.mrg")
			pw = new PrintWriter(out)
			for (treeIndex <- 0 to evalTreeProgression(0).size-1) {
				val mode = evalTreeProgression.map(treeList => treeList(treeIndex)).groupBy(identity).maxBy(_._2.size)._1
				mode.pennPrint(pw)
			}
			pw.close
		}
	}
}

/**
 * An object that sets of the configuration for command-line options using
 * Scallop and returns the options, ready for use.
 */
object LatentPCFGOpts {

  import org.rogach.scallop._
  
  def apply(args: Array[String]) = new ScallopConf(args) {
    banner("""
Tree Sampling using PCFGs with Partial Bracket Information.

For usage see below:
         """)

    val maxSentLength = opt[Int]("max-sentence-length",short='m', default=Some(50), validate = (0<), descr="The maximum sentence length to use.")
    val iterationCount = opt[Int]("iteration-count",short='i', default=Some(5), validate = (0<), descr="The number of training iterations to use.")
	val tagger = opt[String]("tagger",required=false,default=Some(""),descr="POS Tagger Model to use.")
	val unkModel = opt[String]("unk",required=true,descr="UNK Model to use.")
	val unkCutoff = opt[Int]("unkCut", required = false, default = Some(3), descr = "Cutoff for unknown word insertion.")
	val jumpTrees = opt[String]("jumpTrees",required=false,default=Some(""),descr="Location of trees to use for optional jumpstart.")
	val unsup = opt[Boolean]("unsup")
	val pLex = opt[Int]("plex", required = false, default = Some(9999999), descr = "Cutoff for partial Lexicalization.")
	val goldTags = opt[Boolean]("gold")	
	val sm = opt[Double]("smoothing",short='s', default=Some(0.0), descr="Weight for smoothing, default is 0.0")
    val language = opt[String]("language",short='l', default=Some("en"), descr="The language to use (en -> English, zh -> Chinese")
    val help = opt[Boolean]("help", noshort = true, descr = "Show this message")
    val xCount = opt[Int]("unlabeled-nonterminals",short='u', default=Some(0), descr="The number of unlabeled nonTerminals to use. Zero, the default, enables the labeled nonterminal mode.")
    val verbose = opt[Boolean]("verbose")
    val dir = trailArg[String]("directory", descr = "The directory containing the input files.")
    val cky = opt[Boolean]("cky-parsing",short='c',descr="Parse using CKY instead of sampling.")
    val annotations = opt[Int]("nonterminal-annotations",short='h', default=Some(0), descr="The number of annotations to use on the nonterminals. Default is 0.")
    val unevenPrior = opt[Boolean]("uneven-prior",short='x',descr="Parse using an uneven (80/20) prior. Must be used in conjunction with -h 2.")
  }
}

object UnknownClassiferGenerator {
	def main(args: Array[String]): Unit = {
		println("Reading Trees From: "+args(0)) 

		val fis = new FileInputStream(args(0))
        val in = new InputStreamReader(fis, "UTF-8");
        val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
        val norm = new edu.stanford.nlp.trees.BobChrisTreeNormalizer()
        val stripper = par.subcategoryStripper
        val tr = new PennTreeReader(in)

        var trees = List[Tree]()

        var curTree = tr.readTree
        while (curTree != null) {

            trees = trees ++ List(stripper.transformTree(norm.transformTree(curTree)))
            curTree = tr.readTree
        }

        val forms = trees.map(t => List("#","#","#") ++ t.yieldWords.map(w => w.value).toList ++ List("#","#","#"))
        val pos = trees.map(t => List("#","#","#") ++ 
        	t.preOrderNodeList.filter(_.isPreTerminal).map(_.label.value).toList ++
        	List("#","#","#") )

        println("Printing Feature Vectors...")
        val f = new File("featureVectors")
        val pw = new PrintWriter(f)
        for( sentenceIndex <- 0 to forms.size - 1) {
        	val sentence = forms(sentenceIndex)
        	val tags = pos(sentenceIndex)
        	for( wordIndex <- 0 to sentence.size - 1) {
        		if (sentence(wordIndex) != "#") {
        			pw.println( sentence(wordIndex) +" "+ sentence(wordIndex-3) +" "+ sentence(wordIndex-2) +" "+
        						sentence(wordIndex-1) +" "+ sentence(wordIndex+1) +" "+ sentence(wordIndex+2) +" "+ 
        						sentence(wordIndex+3) +" "+ tags(wordIndex-3) +" "+ tags(wordIndex-2) +" "+
        						tags(wordIndex-1) +" "+ tags(wordIndex) +" "+ tags(wordIndex+1) +" "+
        						tags(wordIndex+2) +" "+ tags(wordIndex+3))
        		}
        	}
        }
        pw.close()


		class UnkBatchFeaturizer extends BatchFeaturizer[String,String,String] {
			def apply(examples: Seq[Example[String,String]]) = {
				val attributes = Array("form-3","form-2","form-1","form+1","form+2","form+3",
										"tag-3","tag-2","tag-1","tag", "tag+1","tag+2","tag+3")
				examples.map{ ex =>
					val obs = attributes.zip(ex.features.split("\\s")).map( x => FeatureObservation(x._1+"="+x._2))
					Example(ex.label,obs.toSeq)
				}
			}
		}

		val indexer = new ExampleIndexer(false)
		val raw = readRaw("featureVectors").toList
		val batchFeaturizer = new UnkBatchFeaturizer()
		val examples = batchFeaturizer(raw).map(indexer)
		val (lmap,fmap) = indexer.getMaps; val numFeatures = fmap.size
		val sparseExamples = examples.toIndexedSeq.map { example =>
			example.map { features =>
				SparseVector(numFeatures)(condense(features).map(_.tuple): _*)
			}.features
		}

		println("Clustering...")
		val kmeans = new Kmeans[SparseVector[Double]](
			sparseExamples,
			Kmeans.cosineDistance,
			maxIterations=10
		)

		val (dispersion, centroids) = kmeans.run(args(1).toInt,1)
		val (distance,pred) = kmeans.computeClusterMemberships(centroids)
		
		val feats = scala.io.Source.fromFile("featureVectors").getLines.toList
		for( cluster <- 0 to args(1).toInt - 1) {
			val dir = new File("clusters/"+cluster); dir.mkdirs
			val f = new File("clusters/"+cluster+"/examples")
			val pw = new PrintWriter(f)
			feats.zip(pred).filter(_._2 == cluster).foreach(pw.println)
			pw.close
		}

		println("Building Classifier")
		def fromSingleDirs(topdir: File): Iterator[Example[String,String]] = {
			
			val dirs = topdir.listFiles.filter(_.isDirectory)

			val exampleLists:Array[Iterator[Example[String,String]]] = dirs.map{ d =>
				val file = d.listFiles.head
				for (UnkRE(form,obs) <- scala.io.Source.fromFile(file).getLines)
					yield Example(d.getName, obs)
			}

			exampleLists.foldLeft(Iterator[Example[String,String]]())(_ ++ _)
		}
		val trainingExamples = fromSingleDirs(new File("clusters")).toList
		val config = LiblinearConfig(cost=5.0,eps=0.01)
		val unkFeaturizer = new Featurizer[String,String] {
			def apply(input: String) = {
				val attributes = Array("form-3","form-2","form-1","form+1","form+2","form+3",
										"tag-3","tag-2","tag-1","tag","tag+1","tag+2","tag+3")
				for ((attr,value) <- attributes.zip(input.split("\\s")))
				yield FeatureObservation(attr+"="+value)
			}
		}
		val classifier = trainClassifier(config, unkFeaturizer, trainingExamples)

		// Evaluate
		/*println("Evaluating...")
		val comparisons = for (ex <- fromSingleDirs(new File("clusters")).toList) yield
			(ex.label, classifier.predict(ex.features), ex.features)
		val (goldLabels, predictions, inputs) = comparisons.unzip3
		println(ConfusionMatrix(goldLabels, predictions, inputs))*/
		saveClassifier(classifier,"unk.classifier")
	}
}
