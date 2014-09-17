package latent_pcfg

import java.io._
import edu.stanford.nlp._
import edu.stanford.nlp.trees._
import collection.JavaConversions._
import collection.JavaConverters._
import breeze.stats.distributions._
import edu.stanford.nlp.ling.StringLabel
import breeze.linalg._

import Util._

abstract class Grammar(trees: Treebank, evalTrees: Treebank, tar: String) {

	var t = trees.terminals
	var n = trees.nonTerminals
	var r = trees.productions
	var H = 0
	var targetLabel = tar
	var params = new Array[PCFGParams](n.size)
	var rulesByParent = n.map(nonterm => r.filter(rule => rule._1 == nonterm))
    var thetaIndex = new Array[Int](n.size*n.size*n.size)
    calcThetaIndex

	def updateTheta(bank: String) = {
        params.foreach{ p =>
        	val label = p.label
            val curRules = rulesByParent(n.indexOf(label))
            val prodCount = curRules.size

            // Count Rule Usages
            var alpha = DenseVector.ones[Double](prodCount)
            for( ruleInd <- 0 to curRules.size - 1) {
                val prod = curRules(ruleInd)
                val count = if (bank == "train") trees.countProductions(prod) else evalTrees.countProductions(prod)
		alpha(ruleInd) += count
                alpha(ruleInd) += p.prior(ruleInd)  // Add Prior information
            }

            // Build new Theta
            var theta = Dirichlet(alpha)

            // Update Stored Values
            p.alpha = alpha
            p.theta = theta.sample
        }
	}

	def updateAnnotatedTheta(bank: String) = {
        params.foreach{ p =>
        	val label = p.label
            val curRules = rulesByParent(n.indexOf(label))
            val prodCount = curRules.size

            // Count Rule Usages
            var alpha = DenseVector.ones[Double](prodCount)
            for( ruleInd <- 0 to curRules.size - 1) {
                val prod = curRules(ruleInd)
                val count = if (bank == "train") trees.countAnnotatedProductions(prod) else evalTrees.countAnnotatedProductions(prod)
                alpha(ruleInd) += count
                alpha(ruleInd) += p.prior(ruleInd)  // Add Prior information
            }

            // Build new Theta
            var theta = Dirichlet(alpha)
            
            // Update Stored Values
            p.alpha = alpha
            p.theta = theta.sample
        }
	}

	def resampleTrees = {
		(0 to evalTrees.size-1).foreach{ treeIndex =>
	        val curTree = evalTrees.getTrees(treeIndex).tree
	        val curStr = curTree.yieldWords.mkString(" ")

	        val P = inside_calc(curStr)
	        
	        var success = false
	        var h = 0
            var hCount = 0
	        while ((!success) && h <= H) {
	            val target = new NodeLabel(targetLabel,h.toString)
	            try { 
	                val guess = sample_tree(P,curStr,target,0,curTree.yieldWords.size)
	                evalTrees.getTrees(treeIndex) = new BasicTree(guess)
	                success = true
	            } catch {
	              case e: Exception => {
	                //println(e)
                    val guess = new LabeledScoredTreeNode(new StringLabel(targetLabel))
                    evalTrees.getTrees(treeIndex) = new BasicTree(guess)
                    success = false
                  }
	            }

                hCount += 1
                if (hCount >= 5) {
                    hCount = 0
                    h = h + 1
                }
	        }
	    }
	}

	def inside_calc(sentence:String):Array[DenseMatrix[Double]] = {
        val w = sentence.split(" ").toArray

        // Build Empty Tables
        val P = n.map(nonTerm => new DenseMatrix[Double](w.size+1,w.size+1))
        
        // Base Cases    
        var nonInd = 0
        n.foreach{ nonTerm =>
            for( k <- 1 to w.size) {
                val word = w(k-1)
		
                val curRules = rulesByParent(n.indexOf(nonTerm)).zipWithIndex
                val correctRhs = curRules.filter(x => x._1._2._1._1.equals(word) || x._1._2._2._1.equals(word))
                val thetaInd = if(!correctRhs.isEmpty) correctRhs.head._2 else -1
                
                P(nonInd)(k-1,k) = if(thetaInd < 0) 0.0 else params(nonInd).theta(thetaInd)
            }
            nonInd += 1
        }

        var termlen = 2
        while(termlen <= w.size) {
            var nonIndex = 0
            // For all Nonterminals
            while (nonIndex < n.size) {
                val nonTerm = n(nonIndex)
                var i = 0
                while(i <= (w.size-2) ){
                    var k = i+termlen
                    if(k <= w.size){
                        var tempprob = 0.0
                        //for all possible split position
                        var j = i+1
                        while( j <= (k-1) ) {
                            //for all possible children nodes
                            var bIndex = 0
                            while (bIndex < n.size) {
                                var cIndex = 0
                                while (cIndex < n.size) {                
                                    val thetaInd = thetaIndexLookup(nonIndex,bIndex,cIndex)
                                    if(thetaInd >= 0){
                                        tempprob += (params(nonIndex).theta(thetaInd))*(P(bIndex)(i,j))*(P(cIndex)(j,k))        
                                    }
                                    cIndex += 1
                                }
                                bIndex += 1
                            }
                            j += 1
                        }
                        P(nonIndex)(i,k) = tempprob
                    }
                    i += 1
                }
                nonIndex += 1
            }
            termlen += 1
        }
        return P
    }

    def sample_tree(P:Array[DenseMatrix[Double]],sentence: String,nonTerm: NodeLabel,i:Int,k:Int):Tree = {
        val w = sentence.split(" +").map(w => w+"_T")

        var root = new LabeledScoredTreeNode(new StringLabel(nonTerm.toString))
        val possibleJ = i+1 to k - 1
        val possRules = r.filter(_._1 == nonTerm).toList.zipWithIndex
        val possibleRules = possRules.filter(_._1._2._2._1 != "") // No Unary Rules
        var parentIndex = n.indexOf(nonTerm)

        if (i == k-1) {
            root.setChildren(List(new LabeledScoredTreeNode(new StringLabel(w(k-1).dropRight(2)))))
        } else {
            val a  = DenseVector.ones[Double](possibleJ.size * n.size * n.size)
            val bs = Array.fill(possibleJ.size * n.size * n.size)(new NodeLabel("",""))
            val cs = Array.fill(possibleJ.size * n.size * n.size)(new NodeLabel("",""))
            val js = Array.fill(possibleJ.size * n.size * n.size)(-1)
            var ind = 0
            for( jInd <- 0 to possibleJ.size-1) {
                for( bInd <- 0 to n.size-1) {
                    val B = n(bInd)
                    for( cInd <- 0 to n.size-1) {
                        val C = n(cInd)

                        val thetaInd = thetaIndexLookup(parentIndex,bInd,cInd)

                        val THETA = if (thetaInd < 0) 0.0 else (params(parentIndex).theta(thetaInd))
                        val LEFT = P(bInd)(i,possibleJ(jInd))
                        val RIGHT = P(cInd)(possibleJ(jInd),k)
                        val NORMAL = P(parentIndex)(i,k)
       
                        a(ind) = if(THETA*LEFT*RIGHT != 0.0) {
                                    (THETA*LEFT*RIGHT)/NORMAL
                                } else {
                                    0.0
                                }
                        bs(ind) = B
                        cs(ind) = C
                        js(ind) = possibleJ(jInd)
                        ind += 1
                    }
                }
            }
            val mul = new Multinomial(a)
            val v = mul.sample

            val j = js(v)
            val B = bs(v)
            val C = cs(v)

            val leftChild  = sample_tree(P,sentence,B,i,j)
            val rightChild = sample_tree(P,sentence,C,j,k)
            root.setChildren(List(leftChild,rightChild))
        }
        return root
    }

	def seedAnnotations(h: Int, unevenPrior: Boolean) {
		H = h
        println("Seeding annotations.")
		// Distribute Productions over Annotations
        var newParams   = Set[PCFGParams]()
        var newRules    = Set[Production]()
        var newNonterms = Set[NodeLabel]()
        val hCubed = h*h*h
        val hSquared = h*h

        n.foreach{ nonterm =>   // For All Possible Parent Nodes...
            val curRules = rulesByParent(n.indexOf(nonterm))
            val prodCount = curRules.size
            val alpha = params(n.indexOf(nonterm)).alpha

            // Create new Nonterms
            newNonterms ++= (1 to h).map{anno => (nonterm._1,anno.toString)}.toSet

            (1 to h).foreach{anno =>   // For All Possible Parent Annotations...
                var newAlpha = DenseVector.zeros[Double](prodCount * hSquared)
                var newAlphaInd = 0
                var ruleIndex = 0

                curRules.foreach{ rule =>   // For all rules associated with the current Parent Node...
                    val subProb = alpha(ruleIndex)/hCubed

                    (1 to h).foreach{subAnnoLeft =>        //  For all possible combinations of Child Annotations
                        (1 to h).foreach{subAnnoRight =>   //
                            newRules ++= List(new Production(new NodeLabel(nonterm._1,anno.toString), (new NodeLabel(rule._2._1._1,subAnnoLeft.toString),new NodeLabel(rule._2._2._1,subAnnoRight.toString)) ))
                            
                            if (unevenPrior) {
                                newAlpha(newAlphaInd) = if (anno == 1) (alpha(ruleIndex)*0.8)/hSquared else (alpha(ruleIndex)*0.2)/hSquared
                            } else {
                                newAlpha(newAlphaInd) = alpha(ruleIndex)/hCubed
                            }
							newAlphaInd += 1
                        }
                    }
                    ruleIndex += 1
                }
                newParams += new PCFGParams(new NodeLabel(nonterm._1,anno.toString),newAlpha,Dirichlet(newAlpha).sample,newAlpha)
            }
        }

        n = newNonterms.toArray
        r = newRules.toArray
        rulesByParent = n.map(nonterm => r.filter(rule => rule._1 == nonterm))
        //params = newParams.toArray.map{p => (p.label, p)}.toMap
        params = n.map(nonterm => newParams.filter(p => p.label == nonterm).head)
        calcThetaIndex
	}

	def annotateTrees = {

		var treeIndex = 0
		trees.getTrees.foreach{ t =>

			var tree = t.tree
			val pos = tree.yieldWords.mkString(" ").split(" ")
	        val b = new DenseMatrix[Double](tree.size - tree.yieldWords.size,H)


	        // Build an Indexed Version of the Tree (Node Labels are, e.g., NP~1)
	        var tree_indexAnno = tree.deepCopy
	        var nodeIndex = 0
	        annotate_nodeIndex(tree_indexAnno)
	        def annotate_nodeIndex(node: Tree) {
	            if (node.children.map(_.label.value).mkString(" ").contains("_T")) {
	                // PreTerminal
	                val currentCat = node.label.value
	                node.setLabel(new StringLabel(currentCat+"~"+nodeIndex))
	                nodeIndex += 1
	            } else if (node.children.size == 0) {
	                // Terminal
	            } else {
	                // NonTerminal
	                val currentCat = node.label.value
	                node.setLabel(new StringLabel(currentCat+"~"+nodeIndex))
	                nodeIndex += 1
	                node.children.foreach(annotate_nodeIndex)
	            }
	        }


	        // Build 'b' table
	        annotate_node(tree_indexAnno)
	        def annotate_node(node: Tree) {
	            nodeIndex = node.label.value.split("~")(1).toInt
	            if (node.children.map(_.label.value).mkString(" ").contains("_T")) {
	                // PreTerminal
	                val currentCat = node.label.value
	                for (h_i <- 0 until H) {
	                    b(nodeIndex,h_i) = 1.0
	                }
	            } else if (node.children.size == 0) {
	                // Terminal
	                println("ERROR: Found Terminal: "+node.label.value)
	            } else {
	                // NonTerminal
	                val thisNI  = nodeIndex
	                val leftNI  = node.children.apply(0).label.value.split("~")(1).toInt
	                val rightNI = node.children.apply(1).label.value.split("~")(1).toInt

	                node.children.foreach(annotate_node)

	                for (x <- (1 to H)) {
	                    var sum = 0.0
	                    for (y <- (1 to H)) {
	                        for (z <- (1 to H)) {
	                            val this_label  = new NodeLabel(node.label.value.split("~")(0),x.toString)

	                            //println(node.label.value)
	                            val parentIndex = n.indexOf(this_label)
	                            val leftIndex   = n.indexOf(new NodeLabel(node.children.apply(0).label.value.split("~")(0),y.toString))
	                            val rightIndex  = n.indexOf(new NodeLabel(node.children.apply(1).label.value.split("~")(0),z.toString))

	                            val thetaIndex = thetaIndexLookup(parentIndex,leftIndex,rightIndex)

	                            val b_j = b(leftNI,y-1)
	                            val b_k = b(rightNI,z-1)
	                            try {
	                                val th  = if (thetaIndex == -1) 0.0 else params(n.indexOf(this_label)).theta(thetaIndex)
	                                sum += th * b_j * b_k
	                            } catch {
	                              case e: Exception => {
	                                
	                              }
	                            }
	                            
	                        }
	                    }

	                    b(thisNI,x-1) = sum
	                }
	            }
	        }


	        // Sample Annotations
	        tree_indexAnno.setLabel(new StringLabel(tree_indexAnno.label.value+"<>1"))
	        sample_annotations(tree_indexAnno)
	        def sample_annotations(node: Tree) {

	            if (node.children.size < 2) {
	                node.setLabel(new StringLabel(node.label.value.split("~")(0)+"<>"+node.label.value.split("<>")(1)))
	                return
	            }

	            val labelSplit = node.label.value.split("~")
	            val this_label = new NodeLabel(labelSplit(0),labelSplit(1).split("<>")(1).toString)

	            val pi = DenseVector.zeros[Double](H*H)
	            val b_i = b(labelSplit(1).split("<>")(0).toInt,labelSplit(1).split("<>")(1).toInt - 1)
	            val leftCat  = node.children.apply(0).label.value.split("~")(0)
	            val leftNI   = node.children.apply(0).label.value.split("~")(1).split("<>")(0).toInt
	            val rightCat = node.children.apply(1).label.value.split("~")(0)
	            val rightNI  = node.children.apply(1).label.value.split("~")(1).split("<>")(0).toInt

	            // Build multinomial
	            for( y <- (1 to H)) {
	                for( z <- (1 to H)) {
	                    val parentIndex = n.indexOf(this_label)
	                    val leftIndex   = n.indexOf(new NodeLabel(leftCat,y.toString))
	                    val rightIndex  = n.indexOf(new NodeLabel(rightCat,z.toString))
	                            
	                    val thetaIndex = thetaIndexLookup(parentIndex,leftIndex,rightIndex)
	                    val th  = if (thetaIndex == -1) 0.0 else params(n.indexOf(this_label)).theta(thetaIndex)
	                    val b_j = b(leftNI,y-1)
	                    val b_k = b(rightNI,z-1)
	                    pi((H*(y-1))+(z-1)) = (th*b_j*b_k)/b_i
	                }
	            }

	            // Sample Children Annotation
	            val mul = new Multinomial(pi)
	            val childIndex = mul.sample.toInt

	            val leftAnnotation  = (childIndex / H) + 1
	            val rightAnnotation = (childIndex % H) + 1

	            node.children.apply(0).setLabel(new StringLabel(leftCat+"~"+leftNI+"<>"+leftAnnotation))
	            node.children.apply(1).setLabel(new StringLabel(rightCat+"~"+rightNI+"<>"+rightAnnotation))
	            node.setLabel(new StringLabel(this_label._1+"<>"+this_label._2))

	            node.children.foreach(sample_annotations)
	        }

	        // Copy Annotated Tree
	        trees.getTrees(treeIndex) = new BasicTree(tree_indexAnno)
	        treeIndex += 1
	    }
        
	}

	def calcThetaIndex = {
		// Theta Index Calc
	    println("Calculating ThetaIndex Array.")
	    thetaIndex = new Array[Int](n.size*n.size*n.size)
	    for( parentIndex <- 0 to n.size-1) {
	        val parent = n(parentIndex)
	        val curRules = rulesByParent(parentIndex)
	        for( leftIndex <- 0 to n.size-1) {
	            val left = n(leftIndex)
	            for( rightIndex <- 0 to n.size-1) {
	                val right = n(rightIndex)
	                val thLookupInd = (n.size*n.size)*(parentIndex) + n.size*(leftIndex) + rightIndex

	                thetaIndex(thLookupInd) = curRules.indexOf( (parent,(left,right)) )
	            }
	        }
	    }
	}

	def thetaIndexLookup(ParentIndex:Int, LeftIndex:Int, RightIndex:Int):Int = {
        return thetaIndex((n.size*n.size)*(ParentIndex) + n.size*(LeftIndex) + RightIndex)
    }

    def getTreebank     = trees
    def getEvalTreebank = evalTrees

}

class XGrammar(trees: Treebank, evalTrees: Treebank, numSymbols: Int) extends Grammar(trees,evalTrees,"X1") {
	val numberedNonterms = (1 to numSymbols).map{num => new NodeLabel("X"+num,"0")}

    val numberedR =   for (i <- 0 to numberedNonterms.length - 1) yield {
                            for (j <- 0 to numberedNonterms.length -1) yield {
                                (numberedNonterms(i),numberedNonterms(j))
                            }
                        }
    val numberedT = for( j <- 0 to t.length - 1) yield {
                            (t(j),new NodeLabel("",""))
                        } 
    val numberedRhs = numberedR.flatten ++ numberedT

    val numberedRules = for( i <- 0 to numberedNonterms.length - 1) yield {
                            for( j <- 0 to numberedRhs.length - 1) yield {
                                new Production (numberedNonterms(i),numberedRhs(j))
                            } 
                        }

    n = numberedNonterms.toArray
    r = numberedRules.flatten.toArray
    rulesByParent = n.map(nonterm => r.filter(rule => rule._1 == nonterm))

    println("Calculating X-Grammar Prior...")
    var newParams  = Set[PCFGParams]()
    var priorIndex = 0
    n.foreach{ nonterm =>
        if (priorIndex % 5 == 0) println ("\tNonterminal "+priorIndex+"/"+n.size)
        val curRules = rulesByParent(n.indexOf(nonterm))
        val prodCount = curRules.size
        
        var alpha = DenseVector.ones[Double](prodCount)
        var theta = Dirichlet(alpha)

        newParams += new PCFGParams(nonterm,alpha,theta.sample,alpha)
        priorIndex = priorIndex + 1
    }
    params = n.map(nonterm => newParams.filter(p => p.label == nonterm).head)

}

class SeededGrammar(trees: Treebank, evalTrees: Treebank, tar: String) extends Grammar(trees,evalTrees, tar) {
	println("Calculating Non-Annotated Prior...")
    var priorIndex = 0
    var newParams  = Set[PCFGParams]()
    n.foreach{ nonterm =>
        if (priorIndex % 5 == 0) println ("\tNonterminal "+priorIndex+"/"+n.size)
        val curRules = rulesByParent(n.indexOf(nonterm))
        val prodCount = curRules.size
        
        var alpha = DenseVector.ones[Double](prodCount)

        for( ruleInd <- 0 to prodCount - 1) {
            val prod = curRules(ruleInd)
            val count = trees.countProductions(prod)
            alpha(ruleInd) += count
        }

        var theta = Dirichlet(alpha)
        newParams += new PCFGParams(nonterm,alpha,theta.sample,alpha)
        priorIndex = priorIndex + 1
    }
    params = n.map(nonterm => newParams.filter(p => p.label == nonterm).head)

    def printOutTrees(bank: String, pw: PrintWriter) = {
        evalTrees.getTrees.foreach{bt =>
            val it = bt.tree.iterator
            while(it.hasNext) {
                var curNode = it.next
                var lab = curNode.label.toString.split(",").head
                if (lab.startsWith("(")) lab = lab.drop(1)
                curNode.setLabel(new StringLabel(lab))
            }
            val o = new LabeledScoredTreeNode(new StringLabel("S"))
            o.addChild(bt.tree)
            o.pennPrint(pw)
        }
    }
}


class UnsupervisedGrammar(trainingSentences: Array[String], evalSentences: Array[String],trainingForms: Array[String], evalForms: Array[String], cutoff: Int) {
	var trainTreebank = new Treebank(null)
    var evalTreebank = new Treebank(null)
    var smoothingParameters = collection.mutable.Map[String, Double]()

    println("Building Unsupervised Grammar...")


    // Find forms occurring above pLex threshold
    val rawForms = (trainingForms++evalForms).mkString(" ").split(" ")
    val topWords = rawForms.groupBy(l=>l).map(t => (t._1,t._2.length)).toList
                               .sortBy(_._2).reverse
                               .filter(x => x._2 >= cutoff)
                               .map(x => x._1).toList
    println(topWords)

    // Replace Tags with Lexicalizations for top words
    val convertedTrainingSentences = trainingSentences.zip(trainingForms).map{sentPair =>
        val fs = sentPair._2
        val ts = sentPair._1

        // Gather tags into new TagSentence
        val newTs = fs.split(" ").zip(ts.split(" ")).map{wordPair =>
            
            // Spit out either modified or original POS
            if(topWords.contains(wordPair._1)){
                //println ("Modified Tag: "+wordPair._2+" => "+wordPair._1)
                wordPair._1
            } else {
                wordPair._2
            }
            
        }
        newTs.mkString(" ")
    }


	val pos = (convertedTrainingSentences++evalSentences).mkString(" ").split(" ").toSet.filter(_ != "")
	println("POS Detected: "+pos.size)
	
	//Build list of 'nonterms'/'productions'
	var n = Array[NodeLabel]()
    n ++= Array(new NodeLabel("S","0"))
	pos.foreach{ p =>
		n ++= Array[NodeLabel](new NodeLabel("Y_"+p,"0"),new NodeLabel("L_"+p,"0"),new NodeLabel("R_"+p,"0"))
        smoothingParameters("Y_"+p+"+") = 0.0
        smoothingParameters("+Y_"+p) = 0.0
	}
    var params = new Array[PCFGParams](n.size)
    var rr = Set[Production]()
    pos.foreach{ p =>
        // Y
        val newR = new Production( new NodeLabel("Y_"+p,"0"),
                                    (new NodeLabel("L_"+p,"0"), new NodeLabel("R_"+p,"0"))
                                )
        rr += newR
        // L
        pos.foreach{ newY => 
            val newR = new Production( new NodeLabel("L_"+p,"0"),
                                    (new NodeLabel("Y_"+newY,"0"), new NodeLabel("L_"+p,"0"))
                                )
            rr += newR
        }
        // R
        pos.foreach{ newY => 
            val newR = new Production( new NodeLabel("R_"+p,"0"),
                                    (new NodeLabel("R_"+p,"0"), new NodeLabel("Y_"+newY,"0"))
                                )
            rr += newR
        }
    }
    val r:Array[Production] = rr.toArray

    println("Calculating ThetaIndex Array...")
    var debug = false
    var rulesByParent = n.map(nonterm => r.filter(rule => rule._1 == nonterm))
    var thetaIndex = new Array[Int](n.size*n.size*n.size)
    for( parentIndex <- 0 to n.size-1) {
        debug = false
        val parent = n(parentIndex)
        val curRules = rulesByParent(parentIndex)
        for( leftIndex <- 0 to n.size-1) {
            val left = n(leftIndex)
            for( rightIndex <- 0 to n.size-1) {
                val right = n(rightIndex)
                val thLookupInd = (n.size*n.size)*(parentIndex) + n.size*(leftIndex) + rightIndex

                thetaIndex(thLookupInd) = curRules.indexOf( (parent,(left,right)) )
            }
        }
    }

    def seedTheta() {
        println("Seeding Parameters...")
        var priorIndex = 0
        var newParams  = Set[PCFGParams]()
        n.foreach{ nonterm =>
            if (priorIndex % 5 == 0) println ("\tNonterminal "+priorIndex+"/"+n.size)
            val curRules = rulesByParent(n.indexOf(nonterm))
            val prodCount = curRules.size
            
            var alpha = DenseVector.ones[Double](prodCount)
            var theta = Dirichlet(alpha)
            newParams += new PCFGParams(nonterm,alpha,theta.sample,alpha)
            priorIndex = priorIndex + 1
        }
        params = n.map(nonterm => newParams.filter(p => p.label == nonterm).head)
    }

    def jumpStartTheta(bank: Treebank) = {
        println("Jump Training Theta with Small Set")
        params.foreach{ p =>
            val label = p.label
            val curRules = rulesByParent(n.indexOf(label))
            val prodCount = curRules.size

            // Count Rule Usages
            var alpha = DenseVector.ones[Double](prodCount)
            for( ruleInd <- 0 to curRules.size - 1) {
                val prod = curRules(ruleInd)
                val count = bank.countProductions(prod)
                alpha(ruleInd) += count
                alpha(ruleInd) += p.prior(ruleInd)  // Add Prior information
            }

            // Build new Theta
            var theta = Dirichlet(alpha)

            // Update Stored Values
            p.alpha = alpha
            p.theta = theta.sample
        }
    }

    def updateTheta(bank: String, sm: Double = 0.0) = {
        println("Updating Theta using: "+bank)
        params.foreach{ p =>
            val label = p.label
            val curRules = rulesByParent(n.indexOf(label))
            val prodCount = curRules.size

            // Count Rule Usages
            var alpha = DenseVector.ones[Double](prodCount)
            var smoothing = DenseVector.zeros[Double](prodCount)
            for( ruleInd <- 0 to curRules.size - 1) {
                val prod = curRules(ruleInd)
                val count = if (bank == "train") trainTreebank.countProductions(prod) else evalTreebank.countProductions(prod)
                alpha(ruleInd) += count
                alpha(ruleInd) += p.prior(ruleInd)  // Add Prior information

                if (prod._2._1._1.startsWith("Y")) smoothing(ruleInd) = sm * smoothingParameters(prod._2._1._1+"+")
                else if (prod._2._2._1.startsWith("Y")) smoothing(ruleInd) = sm * smoothingParameters("+"+prod._2._2._1)
            }

            // Build new Theta
            var theta = Dirichlet(alpha)

            // Update Stored Values
            p.alpha = alpha
            p.theta = theta.sample + smoothing
        }
    }

    def resampleTrees():List[Int] = {
        // Sample Trees for provided Sentences (tag sequences)
        val transformedTraining = trainingSentences.map{ s =>
            val words = s.split(" +")
            words.map(w => w+"_L"+" "+w+"_R ").mkString(" ")
        }
        val transformedEval = evalSentences.par.map{ s =>
            val words = s.split(" +")
            words.map(w => w+"_L"+" "+w+"_R ").mkString(" ")
        }

        var badTrees = List(0)
        badTrees = badTrees.drop(1)
        val trainingTrees = transformedTraining.par.map{ sent =>
            var target = "Y_V"
            if (!sent.contains("VBD_")) {
                if (sent.contains("VBG_")) target = "Y_VBG"
                else if (sent.contains("VBZ_")) target = "Y_VBZ"
                else if (sent.contains("VBN_")) target = "Y_VBN"
                else if (sent.contains("VBP_")) target = "Y_VBP"
                else {
                    target = "Y_"+sent.split(" +").head.split("_").head
                    //println("\tNon-verb Target: "+target+" : "+sent)
                }
            }
            try {
                val P = inside_calc(sent)
                val guess = sample_tree(P,sent,(target,"0"),0,sent.split(" +").size)
                guess
            } catch {
              case e: Exception => {
                    badTrees = badTrees ++ List(0)
                    new LabeledScoredTreeNode(new StringLabel("NULL"))   
                }
            }
        }
        if (!badTrees.isEmpty) println("Bad Trees in Training: "+badTrees)
        badTrees = List(0)
        badTrees = badTrees.drop(1)
        val evalTrees = transformedEval.par.map{ sent =>
            var target = "Y_V"
            if (!sent.contains("VBD_")) {
                if (sent.contains("VBG_")) target = "Y_VBG"
                else if (sent.contains("VBZ_")) target = "Y_VBZ"
                else if (sent.contains("VBN_")) target = "Y_VBN"
                else if (sent.contains("VBP_")) target = "Y_VBP"
                else {
                    target = "Y_"+sent.split(" +").head.split("_").head
                    //println("\tNon-verb Target: "+target+" : "+sent)
                }
            }
            try {
                val P = inside_calc(sent)
                val guess = sample_tree(P,sent,(target,"0"),0,sent.split(" +").size)
                guess
            } catch {
              case e: Exception => {
                    badTrees = badTrees ++ List(0)
                    new LabeledScoredTreeNode(new StringLabel("NULL"))   
                }
            }
        }
        if (!badTrees.isEmpty) println("Bad Trees in Eval: "+badTrees)

        // Convert sampled Trees into Treebanks (to facilitate ThetaUpdates)
        trainTreebank = new Treebank(trainingTrees.toArray.map(t => new BasicTree(t)))
        evalTreebank = new Treebank(evalTrees.toArray.map(t => new BasicTree(t)))
        return badTrees
    }

    def calculateSmoothing() {
        smoothingParameters.keys.foreach{ sm =>
            var count = 0
            if (sm.startsWith("+")) { // Right Branching
                val pos = sm.drop(1)
                count = trainTreebank.countSmoothedProductions(false, pos)
            } else {    // Left Branching
                val pos = sm.dropRight(1)
                count = trainTreebank.countSmoothedProductions(true, pos)
            }

            smoothingParameters(sm) = count / trainTreebank.totalProductions.toDouble           
        }
    }

    def inside_calc(sentence:String):Array[DenseMatrix[Double]] = {
        val w = sentence.trim.split(" +").toArray

        // Build Empty Tables
        val P = n.map(nonTerm => new DenseMatrix[Double](w.size+1,w.size+1))
        
        // Base Cases    
        var nonInd = 0
        n.foreach{ nonTerm =>
            for( k <- 1 to w.size) {
                val word = w(k-1)
                val ws=word.split("_")
                
                P(nonInd)(k-1,k) = if(nonTerm == new NodeLabel(ws(1)+"_"+ws(0),"0")) 1.0 else 0.0
                //println(nonTerm+"->"+word+" "+P(nonInd)(k-1,k))
            }
            nonInd += 1
        }

        var termlen = 2
        while(termlen <= w.size) {
            var nonIndex = 0
            // For all Nonterminals
            while (nonIndex < n.size) {
                val nonTerm = n(nonIndex)
                var i = 0
                while(i <= (w.size-2) ){
                    var k = i+termlen
                    if(k <= w.size){
                        var tempprob = 0.0
                        //for all possible split position
                        var j = i+1
                        while( j <= (k-1) ) {
                            //for all possible children nodes
                            var bIndex = 0
                            while (bIndex < n.size) {
                                var cIndex = 0
                                while (cIndex < n.size) {                
                                    val thetaInd = thetaIndexLookup(nonIndex,bIndex,cIndex)
                                    if(thetaInd >= 0){
                                        tempprob += (params(nonIndex).theta(thetaInd))*(P(bIndex)(i,j))*(P(cIndex)(j,k))        
                                    }
                                    cIndex += 1
                                }
                                bIndex += 1
                            }
                            j += 1
                        }
                        P(nonIndex)(i,k) = tempprob
                    }
                    i += 1
                }
                nonIndex += 1
            }
            termlen += 1
        }
        return P
    }

    def sample_tree(P:Array[DenseMatrix[Double]],sentence: String,nonTerm: NodeLabel,i:Int,k:Int):Tree = {
        val w = sentence.split(" +").map(w => w+"_T")

        var root = new LabeledScoredTreeNode(new StringLabel(nonTerm.toString))
        val possibleJ = i+1 to k - 1
        val possRules = r.filter(_._1 == nonTerm).toList.zipWithIndex
        val possibleRules = possRules.filter(_._1._2._2._1 != "") // No Unary Rules
        var parentIndex = n.indexOf(nonTerm)

        if (i == k-1) {
            root.setChildren(List(new LabeledScoredTreeNode(new StringLabel(w(k-1).dropRight(2)))))
        } else {
            val a  = DenseVector.ones[Double](possibleJ.size * n.size * n.size)
            val bs = Array.fill(possibleJ.size * n.size * n.size)(new NodeLabel("",""))
            val cs = Array.fill(possibleJ.size * n.size * n.size)(new NodeLabel("",""))
            val js = Array.fill(possibleJ.size * n.size * n.size)(-1)
            var ind = 0
            for( jInd <- 0 to possibleJ.size-1) {
                for( bInd <- 0 to n.size-1) {
                    val B = n(bInd)
                    for( cInd <- 0 to n.size-1) {
                        val C = n(cInd)

                        val thetaInd = thetaIndexLookup(parentIndex,bInd,cInd)

                        val THETA = if (thetaInd < 0) 0.0 else (params(parentIndex).theta(thetaInd))
                        val LEFT = P(bInd)(i,possibleJ(jInd))
                        val RIGHT = P(cInd)(possibleJ(jInd),k)
                        val NORMAL = P(parentIndex)(i,k)
       
                        a(ind) = if(THETA*LEFT*RIGHT != 0.0) {
                                    (THETA*LEFT*RIGHT)/NORMAL
                                } else {
                                    0.0
                                }
                        bs(ind) = B
                        cs(ind) = C
                        js(ind) = possibleJ(jInd)
                        ind += 1
                    }
                }
            }
            val mul = new Multinomial(a)
            val v = mul.sample

            val j = js(v)
            val B = bs(v)
            val C = cs(v)

            val leftChild  = sample_tree(P,sentence,B,i,j)
            val rightChild = sample_tree(P,sentence,C,j,k)
            root.setChildren(List(leftChild,rightChild))
        }
        return root
    }

    def thetaIndexLookup(ParentIndex:Int, LeftIndex:Int, RightIndex:Int):Int = {
        return thetaIndex((n.size*n.size)*(ParentIndex) + n.size*(LeftIndex) + RightIndex)
    }

    def printOutTrees(bank: String, pw: PrintWriter) = {
        evalTreebank.getTrees.foreach{bt =>
            val it = bt.tree.iterator
            while(it.hasNext) {
                var curNode = it.next
                var lab = curNode.label.toString.split(",").head
                if (lab.startsWith("(")) lab = lab.drop(1)
                curNode.setLabel(new StringLabel(lab))
            }
            val o = new LabeledScoredTreeNode(new StringLabel("S"))
            o.addChild(bt.tree)
            o.pennPrint(pw)
        }
    }
}
