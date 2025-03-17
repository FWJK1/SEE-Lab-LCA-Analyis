# TODOS 

## conceptual and debugging
* improve betweenness centrality filter usefulness -- maybe talk to PSD to get a better ideas of how to use it ...?
* restructure script arrangement? folders?

## multipurpose
* implement the heatmap
* implement the geo-network with new lines for new network

## geolocator
* Improve geolocator logic -- maybe add 'neutral' polygons to select from based on xy of closest connected point? Would probably have to build these in QGIS. Need not use distance, though, could just use ranges. like, cast xy -> int, and then have a dict that just mapped to neutral polygons which they were then placed randomly within

* consider multithreading the calculation

* make the logic more sophisticated for geolocating stuff we do have, per the original notebooks (that is, import the subset matching and so on)

## lca stuff
* At some point we'll have to work with the LCA_analyzer object (as well as change inputs in bload) to work with other LCAs. 

* split apart the bio/techno sphere and the foreground. so that we keep the bio techno always saved and then load in the foreground and alter the main database

* create a mode with just technosphere -- possibly w arguments?

