class FSegment:

    def __init__( self, endCellList, initialPermanence, vector, posRange ):
    # Create a list of synapses with initialPermanence (also stored in a list)
    # Set the vector coordinates for this segment.

        self.synapses    = endCellList
        self.permanences = [ initialPermanence ] * len( endCellList )

        self.dimensions = len( vector )      # Number of dimensions vector positions will be in.
        self.vector     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            posI = ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 )
            self.vector.append( posI )

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                return False

        return True

class FCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False           # Means column burst, or cell was predictive and then column fired.
        self.primed         = False           # This cell is primed by OCells.
        self.predictive     = False           # Means synapses on connected segments above activationThreshold.
        self.segments       = []              # Contains lists of synapses. Each segment attaches to a unique feature.

    def CreateSegment( self, endCellList, initialPermanence, vector, posRange ):
    # Create a new segment with synapses connecting this FCell to all FCells in list, centered around vector given.

        newSegment = FSegment( endCellList, initialPermanence, vector, posRange )
        self.segments.append( newSegment )

class OSynapse:

    def __init__( self, FCell, initialPermanance ):
    # Create a new synapse from Object OCell to Feature FCell.

        self.FCell      = FCell
        self.permanance = initialPermanance

class OCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False
        self.synapses       = []              # Contains lists of synapses. Each synapse primes an FCell.

    def NewSynapse( self, FCell, initialPermanence ):
    # Create a new synapse to prime a FCell.

        newSynapse = OSynapse( FCell, initialPermanence )
        self.synapses.append( newSynapse )
