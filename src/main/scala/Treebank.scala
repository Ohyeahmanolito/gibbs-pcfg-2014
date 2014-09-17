package latent_pcfg

import java.io._
import edu.stanford.nlp._
import edu.stanford.nlp.trees._
import collection.JavaConversions._
import collection.JavaConverters._
import breeze.stats.distributions._
import breeze.linalg._

import Util._

class Treebank(trees: Array[BasicTree]) {
	
	def countProductions(rule: Production):Int = {
		var count = Array.fill(trees.size)(0)
        (0 to trees.size - 1).par.foreach{ treeIndex =>
            count(treeIndex) = trees(treeIndex).countProduction(rule)
        }
        count.sum
	}

	def countAnnotatedProductions(rule: Production):Int = {
		var count = Array.fill(trees.size)(0)
        (0 to trees.size - 1).par.foreach{ treeIndex =>
            count(treeIndex) = trees(treeIndex).countAnnotatedProduction(rule)
        }
        count.sum
	}

	def countSmoothedProductions(left: Boolean, pos:String):Int = {
		var count = Array.fill(trees.size)(0)
        (0 to trees.size - 1).par.foreach{ treeIndex =>
            count(treeIndex) = trees(treeIndex).countSmoothedProduction(left,pos)
        }

        //val l = if (left) "left" else "right"
        //if (count.sum > 0) println("DEBUG: SM-"+l+" "+count.sum+" "+pos)
        count.sum
	}

	def nonTerminals():Array[NodeLabel] = {
		var nonTerminals = Set[NodeLabel]()
		for( ind <- 0 to trees.size - 1) {
	        val treeIt = trees(ind).tree.iterator
	        while (treeIt.hasNext) {
	            val curNode = treeIt.next
	            var childString = curNode.children.toList.map(_.label.value)
	            
	            if (!childString.isEmpty) {
	                nonTerminals ++= Set(new NodeLabel(curNode.label.value,"0"))
	            }
	        }
	    }
	    nonTerminals.toArray
	}

	def terminals():Array[NodeLabel] = {
		var terminals = Set[NodeLabel]()
		for( ind <- 0 to trees.size - 1) {
	        val treeIt = trees(ind).tree.iterator
	        while (treeIt.hasNext) {
	            val curNode = treeIt.next
	            if (curNode.isLeaf) terminals ++= Set(new NodeLabel(curNode.label.value,"0"))
	        }
	    }
	    terminals.toArray
	}

	def productions():Array[Production] = {
		var productions  = Set[Production]()
	    for( ind <- 0 to trees.size - 1) {
	        val treeIt = trees(ind).tree.iterator
	        while (treeIt.hasNext) {
	            val curNode = treeIt.next
	            var childString = curNode.children.toList.map(_.label.value)
	            
	            if (!childString.isEmpty) {
	                if (childString.size == 1) childString = childString ++ List("")
	                val newProd = new Production(new NodeLabel(curNode.label.value,"0"), 
	                                                (new NodeLabel(childString(0),"0"),new NodeLabel(childString(1),"0"))
	                                            )
	                productions += newProd
	            }
	        }
	    }
	    productions.toArray
	}

	def size():Int = trees.size

	def totalProductions():Int = trees.map(t => t.tree.yieldWords.size - 1).sum

	def getTrees = trees

}