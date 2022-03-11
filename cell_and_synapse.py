from bisect import bisect_left
from random import uniform, choice, sample, randrange
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR

class Segment:

    def __init__( self, dimensions, incidentCellsList, terminalCell, vector, vectorCells, initialActivationThreshold,
        initialPermanence, permanenceIncrement, permanenceDecay, numVectorSynapses, vectorRange, vectorSynapseScaleFactor ):
    # Initialize the inidividual segment structure class.

        self.active              = True                         # True means it's predictive, and above threshold terminal cells fired.
        self.timeSinceActive     = 0                            # Time steps since this segment was active last.
        self.activationThreshold = initialActivationThreshold   # Minimum overlap required to activate segment.
        self.activeAboveThresh   = False

        # Synapse portion.
        self.incidentSynapses    = incidentCellsList.copy()
        self.terminalSynapse     = terminalCell

        self.incidentActivation  = []           # A list of all cells that last overlapped with incidentSynapses.

        # Vector portion.
        self.dimensions = dimensions         # Number of dimensions vector positions will be in.
        self.numVectorSynapses = numVectorSynapses
        self.vectorSynapseScaleFactor = vectorSynapseScaleFactor
        self.vectorRange       = []
        for d in range( self.dimensions ):
            self.vectorRange.append( [ -vectorRange, vectorRange ] )
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecay     = permanenceDecay

        if vector != None:
            if len( vector ) != dimensions:
                print( "Vector sent to create segment not of same dimensions sent." )
                exit()

            self.vectorSynapses = []
# MIGHT NEED TO ADD A SCALE FUNCTION, AND RANGE FUNCTION INTO THIS LATER.
            for x in range( self.dimensions ):
                thisDim = [ 0.0 ] * self.numVectorSynapses
                self.vectorSynapses.append( thisDim )
            for y in range( self.dimensions ):
                self.ChangeVectorStrength( y, vector[ y ], initialPermanence )
        else:
            self.vectorSynapses = vectorCells

    def GetVectorSynapseIndex( self, whichDim, position ):
    # Returns the index in self.vectorSynapses for position.

        if position < self.vectorRange[ whichDim ][ 0 ] or position > self.vectorRange[ whichDim ][ 1 ]:
            print( "Sent position to GetVectorSynapseIndex() outside allowed range." )
            exit()

        rangeMin   = self.vectorRange[ whichDim ][ 0 ]
        rangeMax   = self.vectorRange[ whichDim ][ 1 ]
        totalRange = rangeMax - rangeMin

        return int( ( ( position - rangeMin ) / totalRange ) * self.numVectorSynapses )

    def ChangeVectorStrength( self, whichDim, position, permanenceAdjust ):
    # Modifies declared vector position by permanenceAdjust, and smooths this out.

        synIndex = self.GetVectorSynapseIndex( whichDim, position )

        synModif = 0
        thisPermAdjust = permanenceAdjust
        while synIndex + synModif < len( self.vectorSynapses[ whichDim ] ):
            self.vectorSynapses[ whichDim ][ synIndex + synModif ] = ModThisSynapse( self.vectorSynapses[ whichDim ][ synIndex + synModif ], thisPermAdjust, True )
            thisPermAdjust = thisPermAdjust * self.vectorSynapseScaleFactor
            if thisPermAdjust < 0.01:
                break
            synModif += 1

        synModif = 1
        thisPermAdjust = permanenceAdjust * self.vectorSynapseScaleFactor
        while synIndex - synModif >= 0:
            self.vectorSynapses[ whichDim ][ synIndex - synModif ] = ModThisSynapse( self.vectorSynapses[ whichDim ][ synIndex - synModif ], thisPermAdjust, True )
            thisPermAdjust = thisPermAdjust * self.vectorSynapseScaleFactor
            if thisPermAdjust < 0.01:
                break
            synModif += 1

    def Inside( self, vector ):
    # Checks if given vector position is inside range.

        if len( vector ) != self.dimensions:
            print( "Vector sent to Inside() not of same dimensions as Segment." )
            exit()

        for i in range( self.dimensions ):
            synIndex = self.GetVectorSynapseIndex( i, vector[ i ] )
            if self.vectorSynapses[ i ][ synIndex ] == 0.0:
                return False

        for x in range( self.dimensions ):
            # Decay all synapses by a bit.
            for syn in range( self.numVectorSynapses ):
                self.vectorSynapses[ x ][ syn ] = ModThisSynapse( self.vectorSynapses[ x ][ syn ], -self.permanenceDecay, True )

            # Increase the synapses around the vector position.
            self.ChangeVectorStrength( x, vector[ x ], self.permanenceIncrement )

        return True

    def IncidentCellActive( self, incidentCell ):
    # Takes the incident cell and adds it to incidentActivation.

        NoRepeatInsort( self.incidentActivation, incidentCell )

    def CheckActivation( self, vector ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active.

        if len( self.incidentActivation ) >= self.activationThreshold:
            self.activeAboveThresh = True

            if self.Inside( vector ):
                self.active = True
                self.timeSinceActive = 0
                return self.terminalSynapse

        self.active = False
        return None

    def RefreshSegment( self ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.incidentActivation = []
        self.timeSinceActive   += 1
        self.activeAboveThresh  = False

    def RemoveIncidentSynapse( self, synapseToDelete ):
    # Delete the incident synapse sent.

        index = IndexIfItsIn( self.incidentSynapses, synapseToDelete )
        if index != None:
            del self.incidentSynapses[ index ]
        else:
            print( "Attempt to remove synapse from segment, but synapse doesn't exist." )
            exit()

    def NewIncidentSynapse( self, synapseToCreate ):
    # Create a new incident synapse.

        index = bisect_left( self.incidentSynapses, synapseToCreate )

        if index == len( self.incidentSynapses ):
            self.incidentSynapses.append( synapseToCreate )
        elif self.incidentSynapses[ index ] != synapseToCreate:
            self.incidentSynapses.insert( index, synapseToCreate )
        else:
            print( "Attempt to add synapse to segment, but synapse already exists." )
            exit()

    def AlreadySynapseToColumn( self, checkSynapse, cellsPerColumn ):
    # Checks if this segment has a synapse to the same column as checkSynapse.

        for synapse in self.incidentSynapses:
            if int( synapse / cellsPerColumn ) == int( checkSynapse / cellsPerColumn ):
                return True

        return False

    def Equality( self, other, equalityThreshold ):
    # Runs the following comparison of equality: segment1 == segment2, comparing their activation intersection, but not vector.

        if self.terminalSynapse == other.terminalSynapse and len( FastIntersect( self.incidentSynapses, other.incidentSynapses ) ) > equalityThreshold:
            return True
        else:
            return False

    def AdjustThreshold( self, minActivation, maxActivation ):
    # Check the number of bundles that have selected winner. Use this to adjust activationThreshold.

        numWinners = 0

        for incB in self.incidentPermanences:
            if incB == 1.0:
                numWinners += 1

        self.activationThreshold = ( ( minActivation - maxActivation ) * 2 ** ( -1 * numWinners ) ) + maxActivation

    def ReturnVectorCells( self ):
    # Returns the vector cell permanences as a list.

        return self.vectorSynapses.copy()

#-------------------------------------------------------------------------------

class SegmentStructure:

    def __init__( self, vectorDimensions, initialPermanence, permanenceIncrement, permanenceDecrement, permanenceDecay,
        activeThresholdMin, activeThresholdMax, incidentColumnDimensions, incidentCellsPerColumn, maxSynapsesToAddPer,
        maxSynapsesPerSegment, maxTimeSinceActive, equalityThreshold, numVectorSynapses, vectorRange, vectorScaleFactor ):
    # Initialize the segment storage and handling class.

        self.dimensions            = vectorDimensions
        self.initialPermanence     = initialPermanence
        self.permanenceIncrement   = permanenceIncrement
        self.permanenceDecrement   = permanenceDecrement
        self.permanenceDecay       = permanenceDecay
        self.activeThresholdMin    = activeThresholdMin
        self.activeThresholdMax    = activeThresholdMax
        self.incCellsPerColumn     = incidentCellsPerColumn
        self.maxSynapsesToAddPer   = maxSynapsesToAddPer
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxTimeSinceActive    = maxTimeSinceActive
        self.equalityThreshold     = equalityThreshold
        self.numVectorSynapses     = numVectorSynapses
        self.vectorRange           = vectorRange
        self.vectorScaleFactor     = vectorScaleFactor

        self.segments = []                                  # Stores all segments structures.
        self.activeSegments = []

        self.incidentSegments = []                          # Stores the connections from each incident cell to segment.
        for cell in range( incidentCellsPerColumn * incidentColumnDimensions ):
            self.incidentSegments.append( [] )
        self.incidentPermanences = []
        for cell in range( incidentCellsPerColumn * incidentColumnDimensions ):
            self.incidentPermanences.append( [] )

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def AddSynapse( self, incCell, segIndex ):
    # Add a synapse to specified segment.

        insertIndex = bisect_left( self.incidentSegments[ incCell ], segIndex )

        if insertIndex != len( self.incidentSegments[ incCell ] ) and self.incidentSegments[ incCell ][ insertIndex ] == segIndex:
            print( "Synapse to this segment already exists." )
            exit()

        self.incidentSegments[ incCell ].insert( insertIndex, segIndex )
        self.incidentPermanences[ incCell ].insert( insertIndex, self.initialPermanence )
        self.segments[ segIndex ].NewIncidentSynapse( incCell )

    def DeleteSynapse( self, incCell, segIndex ):
    # Remove a synapse to specified segment.

        delIndex = bisect_left( self.incidentSegments[ incCell ], segIndex )

        if delIndex != len( self.incidentSegments[ incCell ] ) and self.incidentSegments[ incCell ][ delIndex ] == segIndex:
            del self.incidentSegments[ incCell ][ delIndex ]
            del self.incidentPermanences[ incCell ][ delIndex ]
            self.segments[ segIndex ].RemoveIncidentSynapse( incCell )

        return delIndex

    def DeleteSegments( self, Cells, segsToDelete ):
    # Receives a list of indices of segments that need deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if len( segsToDelete ) > 0:

            segsToDelete.sort()

            for segIndex in reversed( segsToDelete ):
                # Delete any references to segment, and lower the index of all greater segment reference indices by one.
                for incCell, incList in enumerate( self.incidentSegments ):
                    indexAt = self.DeleteSynapse( incCell, segIndex )

                    while indexAt < len( incList ):
                        incList[ indexAt ] -= 1
                        indexAt += 1

                # Decrease terminal reference.
                Cells[ self.segments[ segIndex ].terminalSynapse ].isTerminalCell -= 1

                # Delete the segment.
                del self.segments[ segIndex ]

    def CreateSegment( self, Cells, incidentCellsList, terminalCell, vector, vectorCells ):
    # Creates a new segment.

        newSegment = Segment( self.dimensions, [], terminalCell, vector, vectorCells, self.activeThresholdMin,
            self.initialPermanence, self.permanenceIncrement, self.permanenceDecay, self.numVectorSynapses,
            self.vectorRange, self.vectorScaleFactor )
        self.segments.append( newSegment )

        indexOfNew = len( self.segments ) - 1
        self.activeSegments.append( indexOfNew )

        for incCell in incidentCellsList:
            self.AddSynapse( incCell, indexOfNew )

        Cells[ terminalCell ].isTerminalCell += 1

    def UnifyVectors( self, vectorCellsList ):
    # Use the list of vector cells contained in vectorCellsList and unify them into one by taking the max for each cell.

        for entry in vectorCellsList:
            if len( entry ) != self.dimensions:
                print( "Dimensions of vectors sent to UnifyVectors() not of correct size." )
                exit()
            for cellList in entry:
                if len( cellList ) != self.numVectorSynapses:
                    print( "Number of cells in vectors sent to UnifyVectors() not of correct size." )
                    exit()

        newVectorCells = []
        for dim in range( self.dimensions ):
            newPermanences = []
            for cell in range( self.numVectorSynapses ):
                max = 0.0
                for entry in vectorCellsList:
                    if entry[ dim ][ cell ] > max:
                        max = entry[ dim ][ cell ]

                newPermanences.append( max )
            newVectorCells.append( newPermanences )

        return newVectorCells

    def UpdateSegmentActivity( self, segsToDelete ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.

        self.activeSegments = []

        for index, segment in enumerate( self.segments ):
            segment.RefreshSegment()
            if segment.timeSinceActive > self.maxTimeSinceActive:
                segsToDelete.append( index )

    def SegmentLearning( self, Cells, lastWinnerCells, doDecayCreate ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        segsToDelete = []

        if doDecayCreate:
            self.DecayAndCreate( Cells, lastWinnerCells )

#        for actSeg in self.activeSegments:
#            segment.AdjustThreshold( self.activeThresholdMin, self.activeThresholdMax )

        self.CheckIfSegsIdentical( Cells, segsToDelete )

        self.UpdateSegmentActivity( segsToDelete )

        self.DeleteSegments( Cells, segsToDelete )

    def GetStimulatedCells( self, activeCells, vector ):
    # Using the activeCells and vector find all segments that activate, and therefore all terminal cells that become
    # stimulated by these segments. Return these stimulated cells as a list.

        stimulatedCells = []

        if len( self.incidentSegments ) > 0:
            # Activate the synapses in segments using activeCells.
            for incCell in activeCells:
                for entry in self.incidentSegments[ incCell ]:
                    self.segments[ entry ].IncidentCellActive( incCell )

            # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
            for segIndex, segment in enumerate( self.segments ):
                cellIfActive = segment.CheckActivation( vector )
                if cellIfActive != None:
                    NoRepeatInsort( self.activeSegments, segIndex )
                    NoRepeatInsort( stimulatedCells, cellIfActive )

        return stimulatedCells

    def ChangePermanence( self, incCell, segIndex, permanenceChange ):
    # Change the permanence of synapse incident on incCell, part of segIndex, by permanenceChange.
    # If permanence == 0.0 then delete it.

        entryIndex = IndexIfItsIn( self.incidentSegments[ incCell ], segIndex )
        if entryIndex != None:
            self.incidentPermanences[ incCell ][ entryIndex ] = ModThisSynapse( self.incidentPermanences[ incCell ][ entryIndex ], permanenceChange, True )

            if self.incidentPermanences[ incCell ][ entryIndex ] <= 0.0:
                self.DeleteSynapse( incCell, segIndex )

    def DecayAndCreate( self, Cells, lastWinnerCells ):
    # For all active segments:
    # 1.) Decrease all synapses on active segments where the terminal cell is not a winner.
    # 2.) Increase synapse strength to active incident cells that already have synapses.
    # 3.) Decrease synapse strength to inactive incident cells that already have synapses.
    # 4.) Build new synapses to active incident winner cells that don't have synapses.

        for activeSeg in self.activeSegments:

            # 1.)...
            if not Cells[ self.segments[ activeSeg ].terminalSynapse ].winner:
                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    self.ChangePermanence( incCell, activeSeg, -self.permanenceDecrement )

            else:
                synapseToAdd = lastWinnerCells.copy()
                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    # 2.)...
                    if Cells[ incCell ].lastActive:
                        self.ChangePermanence( incCell, activeSeg, self.permanenceIncrement )
                    # 3.)...
                    else:
                        self.ChangePermanence( incCell, activeSeg, -self.permanenceDecrement )

                    indexIfIn = IndexIfItsIn( synapseToAdd, incCell )
                    if indexIfIn != None:
                        del synapseToAdd[ indexIfIn ]

                if len( synapseToAdd ) > 0:
                    # 4.)...
                    # Check to make sure this segment doesn't already have a synapse to this column.
                    realSynapsesToAdd = []
                    for synAdd in synapseToAdd:
                        if not self.segments[ activeSeg ].AlreadySynapseToColumn( synAdd, self.incCellsPerColumn ):
                            realSynapsesToAdd.append( synAdd )

                    reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.maxSynapsesToAddPer ) )
                    for toAdd in reallyRealSynapsesToAdd:
                        self.AddSynapse( toAdd, activeSeg )

                    # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
                    while self.maxSynapsesPerSegment - len( self.segments[ activeSeg ].incidentSynapses ) < 0:
                        toDel = choice( self.segments[ activeSeg ].incidentSynapses )
                        self.ChangePermanence( toDel, activeSeg, -1.0 )

    def ReturnActivation( self, seg ):
    # Returns the sum of permanences of synapses incident on incCellList and part of seg.

        activation = 0.0

        for incCell in self.segments[ seg ].incidentActivation:
            entryIndex = IndexIfItsIn( self.incidentSegments[ incCell ], seg )
            if entryIndex != None:
                activation += self.incidentPermanences[ incCell ][ entryIndex ]

        return activation

    def ThereCanBeOnlyOne( self, cellsList ):
    # Return the cell, in cellsList, which is terminal on an active segment with greatest activation.

        greatestActivation = 0.0
        greatestCell       = cellsList[ 0 ]

        if len( cellsList ) > 1:
            for cell in cellsList:
                for actSeg in self.activeSegments:
                    if self.segments[ actSeg ].terminalSynapse == cell:
                        thisActivation = self.ReturnActivation( actSeg )
                        if thisActivation > greatestActivation:
                            greatestActivation = thisActivation
                            greatestCell       = cell

        return ( greatestCell, greatestActivation )

    def CheckIfSegsIdentical( self, Cells, segsToDelete ):
    # Compares all segments to see if they have identical vectors or active synapse bundles. If any do then merge them.
    # A.) Begin by checking which segments have an above threshold overlap activation.
    # B.) Group these segments by forming a list of lists, where each entry is a grouping of segments.
    #   Segments are grouped by first checking their terminal synapse against one, and then checking if their overlap is above equalityThreshold.
    # C.) Then, if any entry has more than two segments in it, merge the two, and mark one of the segments for deletion.

        segmentGroupings = []

        for index, segment in enumerate( self.segments ):
            if segment.activeAboveThresh:
                chosenIndex = None
                for entryIndex, entry in enumerate( segmentGroupings ):
                    thisEntryMatches = True
                    for entrySegment in entry:
                        if not segment.Equality( self.segments[ entrySegment ], self.equalityThreshold ):
                            thisEntryMatches = False
                            break
                    if thisEntryMatches:
                        chosenIndex = entryIndex
                        break

                if chosenIndex == None:
                    segmentGroupings.append( [ index ] )
                else:
                    segmentGroupings[ chosenIndex ].append( index )

        for group in segmentGroupings:
            if len( group ) > 1:
                SDRList     = [ self.segments[ group[ i ] ].incidentSynapses for i in range( len( group) ) ]
                unitySDR    = GenerateUnitySDR( SDRList, len( SDRList[ 0 ] ), self.incCellsPerColumn )
                unityVector = self.UnifyVectors( [ self.segments[ group[ k ] ].ReturnVectorCells() for k in range( len( group ) ) ] )

                self.CreateSegment( Cells, unitySDR, self.segments[ group[ 0 ] ].terminalSynapse, None, unityVector )

                for segIndex in range( len( group ) ):
                    segsToDelete.append( group[ segIndex ] )

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self ):
    # Create a new feature level cell with synapses to OCells.

        # FCell state variables.
        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False

        self.isTerminalCell = 0           # Number of segments this cell is terminal on.

        # For data collection.
        self.activeCount = 0
        self.states      = []
        self.statesCount = []

#    def __repr__( self ):
#    # Returns string properties of the FCell.
#
#        toPrint = []
#        for index in range( len( self.OCellConnections ) ):
#            toPrint.append( ( self.OCellConnections[ index ], self.OCellPermanences[ index ] ) )
#
#        return ( "< ( Connected OCells, Permanence ): %s >"
#            % toPrint )

    def ReceiveStateData( self, stateNumber ):
    # Receive the state number and record it along with the count times, if this cell is active.

        if self.active:
            self.activeCount += 1

            stateIndex = bisect_left( self.states, stateNumber )
            if stateIndex != len( self.states ) and self.states[ stateIndex ] == stateNumber:
                self.statesCount[ stateIndex ] += 1
            else:
                self.states.insert( stateIndex, stateNumber )
                self.statesCount.insert( stateIndex, 1 )

    def ReturnStateInformation( self ):
    # Return the state information and count.

        toReturn = []
        toReturn.append( self.activeCount )
        for i, s in enumerate( self.states ):
            toReturn.append( ( s, self.statesCount[ i ] ) )

        return toReturn

# ------------------------------------------------------------------------------

class OCell:

    def __init__( self ):
    # Create new object cell.

        self.active = False

        self.isTerminalCell = 0

        self.segActivationLevel     = 0
        self.overallActivationLevel = 0.0

    def AddActivation( self, synapseActivation ):
    # Add plust one to segActivationLevel, and add synapseActivation to the overallActivationLevel .

        self.segActivationLevel += 1
        self.overallActivationLevel += synapseActivation

    def ResetState( self ):
    # Make inactive and reset activationLevel.

        self.active = False

        self.segActivationLevel     = 0
        self.overallActivationLevel = 0.0

    def CheckSegmentActivationLevel( self, threshold ):
    # Returns True if segActivationLevel is above threshold.

        if self.segActivationLevel >= threshold:
            return True
        else:
            return False

    def ReturnOverallActivation( self ):
    # Returns self.overallActivationLevel.

        return self.overallActivationLevel
