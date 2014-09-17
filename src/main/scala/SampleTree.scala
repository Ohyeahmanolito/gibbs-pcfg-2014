package latent_pcfg

import java.io._
import edu.stanford.nlp._
import edu.stanford.nlp.trees._
import collection.JavaConversions._
import collection.JavaConverters._
import breeze.stats.distributions._
import breeze.linalg._

import Util._

abstract class SampleTree(source: Tree) {
	var tree = source.deepCopy

	def countProduction(rule: Production):Int

	def sentence() = tree.yieldWords
}

class BasicTree(source: Tree) extends SampleTree(source) {

	def countProduction(rule: Production):Int = {
		var count = 0

       	val treeIt = tree.iterator
       	while(treeIt.hasNext) {
        	val curNode = treeIt.next
            val childString = curNode.children.toList.map(_.label.value).mkString(" ").toUpperCase
            if (childString != "") {
                val prodChildString = rule._2._1._1+" "+rule._2._2._1
                if( (rule._1._1.toString == curNode.label.value) && (childString.trim == prodChildString.trim) ) {
                    count += 1
                }
            }
        }
        return count
	}

	def countAnnotatedProduction(rule: Production):Int = {
		var count = 0

       	val treeIt = tree.iterator
       	while(treeIt.hasNext) {
        	val curNode = treeIt.next
            val childString = curNode.children.toList.map(_.label.value).mkString(" ")
            if (childString != "") {
                val prodChildString = rule._2._1._1+"<>"+rule._2._1._2+" "+rule._2._2._1+"<>"+rule._2._2._2

                if( (rule._1._1+"<>"+rule._1._2 == curNode.label.value) && (childString.trim == prodChildString.trim) ) {
                    count += 1
                }
            }
        }
        return count
	}

    def countSmoothedProduction(left: Boolean, pos: String):Int = {
        var count = 0

        val treeIt = tree.iterator
        while(treeIt.hasNext) {
            val curNode = treeIt.next
            
            if (curNode.children.size == 2) {
                if (left) {
                    if (curNode.getChild(0).label.value.drop(1).dropRight(3).toUpperCase == pos.toUpperCase) count += 1
                } else {
                    if (curNode.getChild(1).label.value.drop(1).dropRight(3).toUpperCase == pos.toUpperCase) count += 1
                }        
            }
        }

        return count
    }
}

class AnnotatedTree(source: Tree) extends SampleTree(source) {

	def countProduction(rule: Production):Int = {
		var count = 0

       	val treeIt = tree.iterator
       	while(treeIt.hasNext) {
        	val curNode = treeIt.next
            val childString = curNode.children.toList.map(_.label.value).mkString(" ")
            if (childString != "") {
                val prodChildString = rule._2._1._1+"<>"+rule._2._1._2+" "+rule._2._2._1+"<>"+rule._2._2._2
                
                if( (rule._1._1+"<>"+rule._1._2 == curNode.label.value) && (childString.trim == prodChildString.trim) ) {
                    count += 1
                }
            }
        }
        return count
	}
}