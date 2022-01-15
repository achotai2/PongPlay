from bisect import bisect_left
from random import uniform, choice, sample
import numpy as np

def BinarySearch( list, val ):
# Search a sorted list: return False if val not in list, and True if it is.

    i = bisect_left( list, val )

    if i != len( list ) and list[ i ] == val:
        return True

    else:
        return False

def IndexIfItsIn( list, val ):
# Returns the left-most index of val in list, if it's there. If val isn't in list then return None.

    i = bisect_left( list, val )

    if i != len( list ) and list[ i ] == val:
        return i

    else:
        return None

def NoRepeatInsort( list, val ):
# Inserts item into list in a sorted way, if it does not already exist (no repeats).
# Returns the index where it inserted val into list.

    idx = bisect_left( list, val )

    if idx == len( list ):
        list.append( val )
        return ( len( list ) - 1 )

    elif list[ idx ] != val:
        list.insert( idx, val )
        return idx

def RepeatInsort( list, val ):
# Inserts item into list in a sorted way (allow repeats).
# Returns the index where it inserted val into list.

    idx = bisect_left( list, val )

    if idx == len( list ):
        list.append( val )
        return ( len( list ) - 1 )

    else:
        list.insert( idx, val )
        return idx

def CheckInside( vector1, vector2, checkRange ):
# Checks if given position is inside range.

    if len( vector1 ) != len( vector2 ):
        print( "Error: Vectors not of same length." )
        exit()

    for i in range( len( vector1 ) ):
        if vector1[ i ] > vector2[ i ] + checkRange or vector1[ i ] < vector2[ i ] - checkRange:
            return False

    return True

#-------------------------------------------------------------------------------

class Segment:

    def __init__( self, terminalCellsList, incidentCellsList, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange, minActivation ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.deleted             = False      # If segment is deleted keep it, but don't look at it until the end report.
        self.active              = False      # True means it's predictive, and above threshold terminal cells fired.
        self.lastActive          = False      # True means it was active last time step.
        self.timeSinceActive     = 0          # Time steps since this segment was active last.
        self.activationThreshold = minActivation
        self.toPrime             = []         # The segments that appear right after this one that might become primed.
        self.toPrimePermanences  = []         # The permanence connections to the toPrime segments.

        # FSegment portion--------
        self.terminalSynapses    = terminalCellsList.copy()
        self.terminalPermanences = [ initialPermanence ] * len( terminalCellsList )

        self.incidentSynapses    = incidentCellsList.copy()
        self.incidentPermanences = [ initialPermanence ] * len( incidentCellsList )

        # Vector portion---------
        self.dimensions = len( vector )      # Number of dimensions vector positions will be in.
        self.vector     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            posI = ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 )
            self.vector.append( posI )

        self.self_record = []

    def AddStateToRecord( self ):
    # Adds details of current state, as a string, to record to be printed upon program completion.

        primedSegString = []
        for index, pSeg in enumerate( self.toPrime ):
            primedSegString.append( ( pSeg, self.toPrimePermanences[ index ] ) )

        return ( "< Vector: %s, \n Active: %s, Last Active: %s, Time Since Active: %s, Activation Threshold: %s \n Active Incident Synapses: %s, %s, %s, \n Active Terminal Synapses: %s, %s, %s > \n"
            % (self.vector, self.active, self.lastActive, self.timeSinceActive, self.activationThreshold, len( self.incidentSynapses ), self.incidentSynapses, self.incidentPermanences, len( self.terminalSynapses ), self.terminalSynapses, self.terminalPermanences ) )

    def GetSegmentData( self ):
    # Return the data collected this time step and reset the segment data.

        if not self.deleted:
            self.self_record.append( self.AddStateToRecord() )

        toReturn = self.self_record.copy()

        self.self_record = []

        return toReturn

    def DeleteSegment( self ):
    # Deletes the segment.

        self.deleted = True
        self.active  = False

        self.self_record.append( "DELETED" )

    def Equality( self, other, equalityThreshold, lowerThreshold ):
    # Runs the following comparison of equality: segment1 == segment2.

        incidentEquality = 0
        terminalEquality = 0

        if self.dimensions == other.dimensions and self.vector == other.vector :

                for index, incSyn in enumerate( self.incidentSynapses ):
                    indexIf = IndexIfItsIn( other.incidentSynapses, incSyn )
                    if indexIf != None and self.incidentPermanences[ index ] >= lowerThreshold and other.incidentPermanences[ indexIf ] >= lowerThreshold:
                            incidentEquality += 1

                for index, terSyn in enumerate( self.terminalSynapses ):
                    indexIf = IndexIfItsIn( other.terminalSynapses, terSyn )
                    if indexIf != None and self.terminalPermanences[ index ] >= lowerThreshold and other.terminalPermanences[ indexIf ] >= lowerThreshold:
                            terminalEquality += 1

        if incidentEquality >= equalityThreshold and terminalEquality >= equalityThreshold:
            self.self_record.append( "EQUALITY True" )
            return True
        else:
            self.self_record.append( "EQUALITY False" )
            return False

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                self.self_record.append( "INSIDE False" )
                return False

        self.self_record.append( "INSIDE True" )
        return True

    def CheckOverlap( self, FCellList, cellsPerColumn, synapsesToCheck, permanencesToCheck, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        self.self_record.append( "CHECK OVERLAP" )

        overlap = 0
        passingCells = []
        failingCells = []

        for index, synapse in enumerate( synapsesToCheck ):
            if FCellList[ synapse ] and permanencesToCheck[ index ] >= lowerThreshold:
                overlap += 1
                passingCells.append( synapse )
            else:
                failingCells.append( synapse )

#        self.self_record.append( "FCell List: " + str( FCellList ) )
        self.self_record.append( "Overlap Score: " + str( overlap ) )
        self.self_record.append( "Passing Overlapping (Col, Cell): " + str( passingCells ) )
        self.self_record.append( "Not Overlapping (Col, Cell): " + str( failingCells ) )
        self.self_record.append( "Activation Threshold: " + str( self.activationThreshold ) )

        if overlap >= self.activationThreshold:
            return True
        else:
            return False

    def CheckIfPredicted( self, activeFCells, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this segments is predicting by checking incident overlap and vector.

        self.self_record.append( "CHECK IF PREDICTED" )

        if self.CheckOverlap( activeFCells, cellsPerColumn, self.incidentSynapses, self.incidentPermanences, lowerThreshold ) and self.Inside( vector ):
            return True
        else:
            return False

    def CheckIfPredicting( self, activeFCells, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this segments is still predicted by checking terminal overlap and vector.

        self.self_record.append( "CHECK IF PREDICTING" )

        if self.CheckOverlap( activeFCells, cellsPerColumn, self.terminalSynapses, self.terminalPermanences, lowerThreshold ) and self.Inside( vector ):
            return True
        else:
            return False

    def ReturnTerminalFCells( self, cellsPerColumn, lowerThreshold ):
    # Return the active terminal cells for this segment in a list format.

        return self.terminalSynapses

    def AdjustThreshold( self, minActivation, maxActivation ):
    # Check the number of bundles that have selected winner. Use this to adjust activationThreshold.

        numWinners = 0

        for terB in self.terminalPermanences:
            if terB == 1.0:
                numWinners += 1
        for incB in self.incidentPermanences:
            if incB == 1.0:
                numWinners += 1

        self.activationThreshold = ( ( minActivation - maxActivation ) * np.exp( -1 * numWinners ) ) + maxActivation
        self.self_record.append( "ADJUST THRESHOLD: " + str( self.activationThreshold ) )

    def DecayAndCreate( self, lastActiveCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn ):
    # Increase synapse strength to lastActive cells that already have synapses.
    # Decrease synapse strength to not lastActive cells that have synapses.
    # Build new synapses to lastActive cells that don't have synapses, but only one cell per column can have synapses.
    # Delete synapses whose permanences have decayed to zero.
    # When adding new synapses make sure keep below maxSynapsesToAddPer and maxSynapsesPerSegment.

        self.self_record.append( "DECAY AND CREATE" )

        synapseToDelete = []
        synapsesToAdd = []

        synIndex = 0
        for cellIndex, lCellActive in enumerate( lastActiveCells ):

            # Increase synapse strength to lastActive cells that already have synapses.
            if lCellActive and synIndex < len( self.incidentSynapses ) and self.incidentSynapses[ synIndex ] == cellIndex:
                self.incidentPermanences[ synIndex ] += permanenceIncrement
                if self.incidentPermanences[ synIndex ] > 1.0:
                    self.incidentPermanences[ synIndex ] = 1.0
                self.self_record.append( "Increase permanence, " + str( cellIndex ) + ", to: " + str( self.incidentPermanences[ synIndex ] ) )
                synIndex += 1

            # Decrease synapse strength to not lastActive cells that have synapses.
            elif not lCellActive and synIndex < len( self.incidentSynapses ) and self.incidentSynapses[ synIndex ] == cellIndex:
                self.incidentPermanences[ synIndex ] -= permanenceDecrement
                if self.incidentPermanences[ synIndex ] < 0.0:
                    synapseToDelete.insert( 0, synIndex )
                self.self_record.append( "Decrease permanence, " + str( cellIndex ) + ", to: " + str( self.incidentPermanences[ synIndex ] ) )
                synIndex += 1

            # Build new synapses to lastActive cells that don't have synapses.
            # Only one cell per column can have synapses, so check for this first.
            # Also need to keep added synapses below maxSynapsesToAddPer and maxSynapsesPerSegment.
            elif lCellActive:
                thisColumn = int( cellIndex / cellsPerColumn )
                addNewSynapse = False

                if synIndex == 0:
                    if int( self.incidentSynapses[ synIndex ] / cellsPerColumn ) != thisColumn:
                        addNewSynapse = True

                elif synIndex < len( self.incidentSynapses ):
                    if int( self.incidentSynapses[ synIndex ] / cellsPerColumn ) != thisColumn and int( self.incidentSynapses[ synIndex - 1 ] / cellsPerColumn ) != thisColumn:
                        addNewSynapse = True

                elif synIndex == len( self.incidentSynapses ):
                    if int( self.incidentSynapses[ synIndex - 1 ] / cellsPerColumn ) != thisColumn:
                        addNewSynapse = True

                if addNewSynapse:
                    synapsesToAdd.append( cellIndex )
                    synIndex += 1

        # Delete synapses whose permanences have decayed to zero.
        for toDel in synapseToDelete:
            del self.incidentSynapses[ toDel ]
            del self.incidentPermanences[ toDel ]

        realSynapsesToAdd = sample( synapsesToAdd, min( len( synapsesToAdd ), maxSynapsesToAddPer, maxSynapsesPerSegment - len( self.incidentSynapses ) ) )
        self.self_record.append( "Added new synapses to: " + str( realSynapsesToAdd ) )
        for toAdd in realSynapsesToAdd:
            index = bisect_left( self.incidentSynapses, toAdd )
            self.incidentSynapses.insert( index, toAdd )
            self.incidentPermanences.insert( index, toAdd )

# -----------------------------------------------------------------------------------

class FCell:

    def __init__( self, initialPermanence, numOCells, pctAllowedOCellConns ):
    # Create a new feature level cell with synapses to OCells.

        self.active     = False
        self.lastActive = False
        self.predictive = False
        self.winner     = False
        self.lastWinner = False

        self.numWinners = 0

        # Create synapse connections to OCells.
        numberOCellConns = int( pctAllowedOCellConns * numOCells )
        self.OCellConnections = sorted( sample( range( numOCells ), numberOCellConns ) )
        self.OCellPermanences = []
        for i in range( numberOCellConns ):
            self.OCellPermanences.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )

    def __repr__( self ):
    # Returns string properties of the FCell.

        toPrint = []
        for index in range( len( self.OCellConnections ) ):
            toPrint.append( ( self.OCellConnections[ index ], self.OCellPermanences[ index ] ) )

        return ( "< ( Connected OCells, Permanence ): %s >"
            % toPrint )

    def OCellConnect( self, totalActiveOCells, permanenceIncrement, permanenceDecrement ):
    # Look through the ordered indices of totalOCellConnections. Choose the lowest one in our OCellConnections
    # to remove, and the highest one not in our OCellConnections to add.

        for index, oCell in enumerate( self.OCellConnections ):
            if BinarySearch( totalActiveOCells, oCell ):
                self.OCellPermanences[ index ] += permanenceIncrement

                if self.OCellPermanences[ index ] > 1.0:
                    self.OCellPermanences[ index ] = 1.0
            else:
                self.OCellPermanences[ index ] -= permanenceDecrement

                if self.OCellPermanences[ index ] < 0.0:
                    self.OCellPermanences[ index ] = 0.0
