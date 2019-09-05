#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    List of alias for classifiers available on python-weka-wrapper.
    
    Name: weka_alias.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from interface.interface import InterfaceException as IException

_weka_alias = {    
    
    ### Bayes 
    "AODE": "weka.classifiers.bayes.AODE",
    "AODEsr": "weka.classifiers.bayes.AODEsr",
    "BayesianLogisticRegression": "weka.classifiers.bayes.BayesianLogisticRegression",
    "BayesNet": "weka.classifiers.bayes.BayesNet",
    "ComplementNaiveBayes": "weka.classifiers.bayes.ComplementNaiveBayes",
    "DMNBtext": "weka.classifiers.bayes.DMNBtext",
    "HNB": "weka.classifiers.bayes.HNB",
    "NaiveBayes": "weka.classifiers.bayes.NaiveBayes",
    "NaiveBayesMultinomial": "weka.classifiers.bayes.NaiveBayesMultinomial",
    "NaiveBayesMultinomialUpdateable": "weka.classifiers.bayes.NaiveBayesMultinomialUpdateable",
    "NaiveBayesMultinomialText": "weka.classifiers.bayes.NaiveBayesMultinomialText",
    "NaiveBayesSimple": "weka.classifiers.bayes.NaiveBayesSimple",
    "NaiveBayesUpdateable": "weka.classifiers.bayes.NaiveBayesUpdateable",
    "WAODE": "weka.classifiers.bayes.WAODE",

    ### Functions
    "GaussianProcesses": "weka.classifiers.functions.GaussianProcesses",
    "IsotonicRegression": "weka.classifiers.functions.IsotonicRegression",
    "LeastMedSq": "weka.classifiers.functions.LeastMedSq",
    "LibLINEAR": "weka.classifiers.functions.LibLINEAR",
    "LibSVM": "weka.classifiers.functions.LibSVM",
    "LinearRegression": "weka.classifiers.functions.LinearRegression",
    "Logistic": "weka.classifiers.functions.Logistic",
    "MLPClassifier": "weka.classifiers.functions.MLPClassifier",
    "MLPRegressor": "weka.classifiers.functions.MLPRegressor",
    "MultilayerPerceptron": "weka.classifiers.functions.MultilayerPerceptron",
    "NonNegativeLogisticRegression": "weka.classifiers.functions.NonNegativeLogisticRegression",
    "PaceRegression": "weka.classifiers.functions.PaceRegression",
    "PLSClassifier": "weka.classifiers.functions.PLSClassifier",
    "RBFNetwork": "weka.classifiers.functions.RBFNetwork",
    "RBFClassifier": "weka.classifiers.functions.RBFClassifier",
    "RBFRegressor": "weka.classifiers.functions.RBFRegressor",
    "SimpleLinearRegression": "weka.classifiers.functions.SimpleLinearRegression",
    "SimpleLogistic": "weka.classifiers.functions.SimpleLogistic",
    "SGD": "weka.classifiers.functions.SGD",
    "SGDText": "weka.classifiers.functions.SGDText",
    "SMO": "weka.classifiers.functions.SMO",
    "SMOreg": "weka.classifiers.functions.SMOreg",
    "SPegasos": "weka.classifiers.functions.SPegasos",
    "SVMreg": "weka.classifiers.functions.SVMreg",
    "VotedPerceptron": "weka.classifiers.functions.VotedPerceptron",
    "Winnow": "weka.classifiers.functions.Winnow",
    
    ### Lazy
    "IB1": "weka.classifiers.lazy.IB1",
    "IBk": "weka.classifiers.lazy.IBk",
    "KStar": "weka.classifiers.lazy.KStar",
    "LBR": "weka.classifiers.lazy.LBR",
    "LWL": "weka.classifiers.lazy.LWL",
    
    # Meta
    "AdaBoostM1": "weka.classifiers.meta.AdaBoostM1",
    "AdditiveRegression": "weka.classifiers.meta.AdditiveRegression",
    "AttributeSelectedClassifier": "weka.classifiers.meta.AttributeSelectedClassifier",
    "Bagging": "weka.classifiers.meta.Bagging",
    "ClassificationViaClustering": "weka.classifiers.meta.ClassificationViaClustering",
    "ClassificationViaRegression": "weka.classifiers.meta.ClassificationViaRegression",
    "CostSensitiveClassifier": "weka.classifiers.meta.CostSensitiveClassifier",
    "CVParameterSelection": "weka.classifiers.meta.CVParameterSelection",
    "Dagging": "weka.classifiers.meta.Dagging",
    "Decorate": "weka.classifiers.meta.Decorate",
    "END": "weka.classifiers.meta.END",
    "EnsembleSelection": "weka.classifiers.meta.EnsembleSelection",
    "FilteredClassifier": "weka.classifiers.meta.FilteredClassifier",
    "Grading": "weka.classifiers.meta.Grading",
    "GridSearch": "weka.classifiers.meta.GridSearch",
    "LogitBoost": "weka.classifiers.meta.LogitBoost",
    "MetaCost": "weka.classifiers.meta.MetaCost",
    "MultiBoostAB": "weka.classifiers.meta.MultiBoostAB",
    "MultiClassClassifier": "weka.classifiers.meta.MultiClassClassifier",
    "MultiClassClassifierUpdateable": "weka.classifiers.meta.MultiClassClassifierUpdateable",
    "MultiScheme": "weka.classifiers.meta.MultiScheme",
    "OneClassClassifier": "weka.classifiers.meta.OneClassClassifier",
    "OrdinalClassClassifier": "weka.classifiers.meta.OrdinalClassClassifier",
    "RacedIncrementalLogitBoost": "weka.classifiers.meta.RacedIncrementalLogitBoost",
    "RandomCommittee": "weka.classifiers.meta.RandomCommittee",
    "RandomSubSpace": "weka.classifiers.meta.RandomSubSpace",
    "RealAdaBoost": "weka.classifiers.meta.RealAdaBoost",
    "RegressionByDiscretization": "weka.classifiers.meta.RegressionByDiscretization",
    "RotationForest": "weka.classifiers.meta.RotationForest",
    "Stacking": "weka.classifiers.meta.Stacking",
    "StackingC": "weka.classifiers.meta.StackingC",
    "ThresholdSelector": "weka.classifiers.meta.ThresholdSelector",
    "Vote": "weka.classifiers.meta.Vote",
    "ClassBalancedND": "weka.classifiers.meta.ClassBalancedND",
    "DataNearBalancedND": "weka.classifiers.meta.DataNearBalancedND",
    "ND": "weka.classifiers.meta.ND",

    ### Miscellaneous
    "HyperPipes": "weka.classifiers.misc.sc.HyperPipes",
    "InputMappedClassifier": "weka.classifiers.misc.sc.InputMappedClassifier",
    "MinMaxExtension": "weka.classifiers.misc.sc.MinMaxExtension",
    "OLM": "weka.classifiers.misc.sc.OLM",
    "OSDL": "weka.classifiers.misc.sc.OSDL",
    "SerializedClassifier": "weka.classifiers.misc.sc.SerializedClassifier",
    "VFI": "weka.classifiers.misc.sc.VFI",
    
    ### Multi-instance
    "CitationKNN": "weka.classifiers.mi.CitationKNN",
    "MDD": "weka.classifiers.mi.MDD",
    "MIBoost": "weka.classifiers.mi.MIBoost",
    "MIDD": "weka.classifiers.mi.MIDD",
    "MIEMDD": "weka.classifiers.mi.MIEMDD",
    "MILR": "weka.classifiers.mi.MILR",
    "MINND": "weka.classifiers.mi.MINND",
    "MIOptimalBall": "weka.classifiers.mi.MIOptimalBall",
    "MIRI": "weka.classifiers.mi.MIRI",
    "MISMO": "weka.classifiers.mi.MISMO",
    "MISVM": "weka.classifiers.mi.MISVM",
    "MITI": "weka.classifiers.mi.MITI",
    "MIWrapper": "weka.classifiers.mi.MIWrapper",
    "SimpleMI": "weka.classifiers.mi.SimpleMI",
    "TLD": "weka.classifiers.mi.TLD",
    "TLDSimple": "weka.classifiers.mi.TLDSimple",

    ### Rules
    "ConjunctiveRule": "weka.classifiers.rules.ConjunctiveRule",
    "DecisionTable": "weka.classifiers.rules.DecisionTable",
    "DTNB": "weka.classifiers.rules.DTNB",
    "FURIA": "weka.classifiers.rules.FURIA",
    "JRip": "weka.classifiers.rules.JRip",
    "M5Rules": "weka.classifiers.rules.M5Rules",
    "NNge": "weka.classifiers.rules.NNge",
    "OneR": "weka.classifiers.rules.OneR",
    "PART": "weka.classifiers.rules.PART",
    "Prism": "weka.classifiers.rules.Prism",
    "Ridor": "weka.classifiers.rules.Ridor",
    "ZeroR": "weka.classifiers.rules.ZeroR",
    
    ### Trees
    "ADTree": "weka.classifiers.trees.ADTree",
    "BFTree": "weka.classifiers.trees.BFTree",
    "DecisionStump": "weka.classifiers.trees.DecisionStump",
    "ExtraTree": "weka.classifiers.trees.ExtraTree",
    "FT": "weka.classifiers.trees.FT",
    "HoeffdingTree": "weka.classifiers.trees.HoeffdingTree",
    "Id3": "weka.classifiers.trees.Id3",
    "J48": "weka.classifiers.trees.J48",
    "J48graft": "weka.classifiers.trees.J48graft",
    "LADTree": "weka.classifiers.trees.LADTree",
    "LMT": "weka.classifiers.trees.LMT",
    "M5P": "weka.classifiers.trees.M5P",
    "NBTree": "weka.classifiers.trees.NBTree",
    "RandomForest": "weka.classifiers.trees.RandomForest",
    "RandomTree": "weka.classifiers.trees.RandomTree",
    "REPTree": "weka.classifiers.trees.REPTree",
    "SimpleCart": "weka.classifiers.trees.SimpleCart",
    "UserClassifier": "weka.classifiers.trees.UserClassifier"

}

class WekaAlias(object): 
    """Class of alias for classifiers available on python-weka-wrapper."""
    
    @staticmethod
    def get_classifier(name):        
        """Return full name of classifier.
        
        Parameters
        ----------
        name : string
            Alias of classifier, not case sensitive.
            
        Returns
        -------
        classifier : string
            Full name of classifier.
        
        Raises
        ------
        IException 'Invalid classifier'
            The user must install the required dependencies to classifiers.
        """
        classifiers = [_weka_alias[alias] for alias in _weka_alias]
        
        if name in classifiers:
            return name
        
        alias = name.upper().strip()
        aliases = [a.upper() for a in _weka_alias]
        
        if alias in aliases:
            return _weka_alias.values()[ aliases.index(alias) ]
        
        raise IException('Invalid classifier')


    @staticmethod
    def get_aliases():
        """Return all aliases.
            
        Returns
        -------
        classifier : Dictionary
            Dictionary with all aliases.
        """
        return _weka_alias
