# TODOS 

* Fix the geo_viz app in the modular setup -- put all the logic in geo_plot, and split it up into separate functions for each trace.

* add functionality to filter on betweeness centrality and streamline / fix filtering functionality generally

* Improve geolocator logic -- maybe add 'neutral' polygons to select from based on xy of closest connected point? Would probably have to build these in QGIS. Need not use distance, though, could just use ranges. like, cast xy -> int, and then have a dict that just mapped to neutral polygons which they were then placed randomly within

* also, make the logic more sophisticated for geolocating stuff we do have, per the original notebooks (that is, import the subset matching and so on)

* At some point we'll have to work with the LCA_analyzer object (as well as change inputs in bload) to work with other LCAs. 

* go back through and set all the self.function methods to return stuff instead of setting self. that will make them easier to test independently. 

* constructing tests would be nice
