   
#################################################################################
#VISUALIZER CODE
#this forms part of the view portion of the MVC
#used for exploring and viewing different parts of data and results from the analyzer. 
#display results and pandas data frames in command line, notebook or as a flask API service. 
class RiverbedVisualizer:
  def __init__(self, project_name, analyzer, searcher_indexer, processor):
    self.project_name, self.analyzer, self.searcher_indexer, self.processor = project_name, analyzer, searcher_indexer, processor
  
  def draw(self):
    pass
  
  def df(self, slice):
    pass
  
