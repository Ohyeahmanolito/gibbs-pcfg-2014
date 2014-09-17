package latent_pcfg

import java.io._
import edu.stanford.nlp._
import edu.stanford.nlp.trees._
import collection.JavaConversions._
import collection.JavaConverters._
import breeze.stats.distributions._
import edu.stanford.nlp.ling.StringLabel
import breeze.linalg._
import nak.NakContext._
import nak.core._
import nak.data._
import nak.cluster._

object Util {

    type NodeLabel  = (String,String)
    type Production = (NodeLabel,(NodeLabel,NodeLabel))

    class PCFGParams(lab:NodeLabel,alph:DenseVector[Double],thet:DenseVector[Double],pri:DenseVector[Double]) {
        
        val label = lab
        var alpha = alph
        var theta = thet
        val prior = pri

        override def toString(): String =  {
            "Params ("+label+"): "+theta.size+":"+theta
        }
    }

	def read_sentences(dir: String, maxSize: Int):(Array[String],Array[String]) = {
		println("Loading Sentences for unsupervised mode...")

		val trainSource = scala.io.Source.fromFile(dir+"/training/train.sentences")
		val trainLines = trainSource.getLines mkString "\n"
		val evalSource = scala.io.Source.fromFile(dir+"/testing/dev.sentences")
		val evalLines = evalSource.getLines mkString "\n"
		trainSource.close(); evalSource.close()

		val trainSents = trainLines.split("\n").filter(s => s.split(" ").size <= maxSize)
		val evalSents  = evalLines.split("\n").filter(s => s.split(" ").size <= maxSize)

		return (trainSents,evalSents)
	}

    def read_trees(dir: String, lang: String, maxSize: Int):(Array[BasicTree],Array[BasicTree]) = {
        println("Loading Trees...")
        var trees = scala.collection.mutable.IndexedSeq[Tree]()
        var evalTrees = scala.collection.mutable.IndexedSeq[Tree]()

        if (lang != "zh") {
            // Load Train-WSJ
            read_wsj(dir+"/training/train-bin.mrg") match {
                case Some(tr) => {
                    trees = trees ++ tr
                }
                case None => println("Couldn't Load Trees!")
            }
            // Load Test-WSJ
            read_wsj(dir+"/testing/dev-bin.mrg") match {
                case Some(tr) => {
                    evalTrees = evalTrees ++ tr
                }
                case None => println("Couldn't Load Trees!")
            }
        } else {
            // Load Train-CHTB
            read_chtb(dir+"/training/train-bin.fid") match {
                case Some(tr) => {
                    trees = trees ++ tr
                }
                case None =>
            }
            // Load Test-CHTB
            read_chtb(dir+"/testing/dev-bin.fid") match {
                case Some(tr) => {
                    evalTrees = evalTrees ++ tr
                }
                case None =>
            }
        }
        (trees.filter(t => t.yieldWords.size < maxSize).toArray.map(x => new BasicTree(x)),
            evalTrees.filter(t => t.yieldWords.size < maxSize).toArray.map(x => new BasicTree(x)))
    }
    
    def read_wsj(filename: String):Option[IndexedSeq[Tree]] = {
        if(!new java.io.File(filename).exists) return None
        val fis = new FileInputStream(filename)
        val in = new InputStreamReader(fis, "UTF-8");
        val par = new edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams
        val norm = new edu.stanford.nlp.trees.BobChrisTreeNormalizer()
        val stripper = par.subcategoryStripper
        val tr = new PennTreeReader(in)
        
        var trees = List[Tree]()

        var curTree = tr.readTree
        while (curTree != null) {

            trees = trees ++ List(CNF(CNF(stripper.transformTree(norm.transformTree(curTree)))))
            curTree = tr.readTree
        }

        return Some(trees.toIndexedSeq.filter(t => t.label.value.startsWith("S")))
    }

    def read_chtb(filename: String):Option[IndexedSeq[Tree]] = {
        if(!new java.io.File(filename).exists) return None
        val fis = new FileInputStream(filename)
        val in = new InputStreamReader(fis, "UTF-8");
        val tf = new edu.stanford.nlp.trees.LabeledScoredTreeFactory
        val norm = new edu.stanford.nlp.trees.BobChrisTreeNormalizer()
        val tok = new edu.stanford.nlp.trees.international.pennchinese.CHTBTokenizer(in)
        val tr = new edu.stanford.nlp.trees.international.pennchinese.FragDiscardingPennTreeReader(in,tf,norm,tok)
        val par = new edu.stanford.nlp.parser.lexparser.ChineseTreebankParserParams
        val stripper = par.subcategoryStripper
        
        var trees = List[Tree]()

        var curTree = tr.readTree
        while (curTree != null) {
            trees = trees ++ List(CNF(CNF(stripper.transformTree(norm.transformTree(curTree)))))
            curTree = tr.readTree
        }

        return Some(trees.toIndexedSeq.filter(t => t.label.value.startsWith("S")))
    }

    def CNF(curTree: Tree):Tree = {
        var toProcess = Set(curTree)
        while(!toProcess.isEmpty) {
            var curNode = toProcess.head
            toProcess = toProcess &~ Set(curNode)
            if (curNode.children.size == 1) {
                if (curNode.getChild(0).isLeaf) {
                    // Lexical Rule
                    if (!curNode.getChild(0).label.value.endsWith("_T")) {
                        curNode.getChild(0).setLabel(new StringLabel(curNode.getChild(0).label.value+"_T"))
                    }
                } else {
                    // Unary Rule that isn't lexical -- Fix it.
                    curNode.setLabel(curNode.getChild(0).label)
                    curNode.setChildren(curNode.getChild(0).children.toList)
                }
            } else if (curNode.children.size == 2) {
                // Binary Rule
            } else if (curNode.children.size > 2) {
                // N-Ary Rule -- Fix it.
                val labelString = "X"//curNode.children.drop(1).toList.map(c => c.nodeString).mkString("-")
                var newNode = new LabeledScoredTreeNode(new StringLabel(labelString))
                newNode.setChildren(curNode.children.drop(1).toList)
                curNode.setChildren(List(curNode.getChild(0),newNode))
            }
            toProcess ++= curNode.children.toSet
        }
        return curTree
    }

    def POS(curTree: Tree):Tree = {
        var toProcess = Set(curTree)
        while(!toProcess.isEmpty) {
            var curNode = toProcess.head
            toProcess = toProcess &~ Set(curNode)
            if (curNode.children.size == 1) {
                if (curNode.getChild(0).isLeaf)
                    curNode.setChildren(List(new LabeledScoredTreeNode(new StringLabel(curNode.nodeString+"_T"))))
            }
            toProcess ++= curNode.children.toSet
        }
        return curTree
    }

    val UnkRE = """^(\S+)\s(.*)$""".r
        def readRaw(filename: String) =
            for (UnkRE(form,obs) <- scala.io.Source.fromFile(filename).getLines)
            yield Example(form, obs)
}
