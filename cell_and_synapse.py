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

class SynapseBundle:

    def __init__( self, cellsInBundle, initialPermanence ):
    # Create a new synapse bundle to all cells in a particular column, with dead synapses if initialPermanence = 0.0.
    # Otherwise bundle is set to active, and permanences are random.

        self.chosenWinner = False

        if initialPermanence == 0.0:
            self.active   = False                       # Active if any cell in bundle has positive permanence.
            self.synapses = [ 0.0 ] * cellsInBundle     # All permanence values initially set to zero.

        elif initialPermanence > 0.0:
            self.active = True
            self.synapses = []
            for index in range( cellsInBundle ):
                self.synapses.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )

        else:
            print( "initialPermanence must be positive or 0.0, not negative." )
            exit()

#    def __eq__( self, other ):
#    # Runs if compared SynapseBundle1 == SynapseBundle2. Compares the permanence strength of each synapse in bundle.
#
#        if len( self.synapses ) == len( other.synapses ) and self.active and other.active:
#
#            for i in range( len( self.synapses ) ):
#                if self.synapses[ i ] != other.synapses[ i ]:
#                    return False
#
#            return True
#
#        else:
#            return False

    def CreateRandomSynapses( self, initialPermanence ):
    # Set bundle to active and create random synapse permanences.

        self.active = True

        for index in range( len( self.synapses ) ):
            self.synapses[index] = uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence )

    def SynapseDecay( self, permanenceDecay ):
    # Decay all synapses in bundle by amount.
    # If synapse is at 1.0 then it is locked in, don't decay it.

        self.active = False

        for index in range( len( self.synapses ) ):
            if self.synapses[ index ] < 1.0:
                self.synapses[ index ] -= permanenceDecay

            if self.synapses[ index ] <= 0.0:
                self.synapses[ index ] = 0.0
            else:
                self.active = True

        return self.active

    def getWinnerCell( self ):
    # Return the cell with the highest synapse and its permanence value, as a tuple.

        winnerCell       = 0
        winnerPermanence = 0.0

        for index in range( len( self.synapses ) ):
            if self.synapses[ index ] >= winnerPermanence:
                winnerCell       = index
                winnerPermanence = self.synapses[ index ]

        return ( winnerCell, winnerPermanence )

    def SupportWinner( self, winnerCell, permanenceIncrement, permanenceDecrement ):
    # Support the winner cells permanences, and decay the loser cells.

        for index in range( len( self.synapses ) ):
            if index == winnerCell:
                if self.synapses[ index ] < 1.0 - permanenceIncrement:
                    self.synapses[ index ] += permanenceIncrement
                else:
                    self.synapses[ index ] = 1.0
                    self.chosenWinner = True
            else:
                if self.synapses[ index ] > 0.0 + permanenceDecrement:
                    self.synapses[ index ] -= permanenceDecrement
                else:
                    self.synapses[ index ] = 0.0

    def ReturnActiveCells( self, lowerThreshold ):
    # Return a list of the active cells.

        cellList = []

        for index in range( len( self.synapses ) ):
            if self.synapses[ index ] > lowerThreshold:
                cellList.append( index )

        return cellList

    def ReturnTrueIfActive( self, index, lowerThreshold ):
    # Returns true if cell index is active.

        if self.synapses[ index ] > lowerThreshold:
            return True
        else:
            return False

    def DeleteBundle( self ):
    # Sets all permanences in this bundle to 0.0 and the bundle to not active.

        self.active       = False
        self.chosenWinner = False

        for index in range( len( self.synapses ) ):
            self.synapses[ index ] = 0.0

class Segment:

    def __init__( self, terminalColumnsList, incidentColumnsList, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange, minActivation ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.deleted             = False      # If segment is deleted keep it, but don't look at it until the end report.
        self.active              = False      # True means it's predictive, and above threshold terminal cells fired.
        self.lastActive          = False      # True means it was active last time step.
        self.timeSinceActive     = 0          # Time steps since this segment was active last.
        self.numWinners          = 0          # The number of synapse bundles that have selected a winner.
        self.activationThreshold = minActivation
        self.toPrime             = []         # The segments that appear right after this one that might become primed.
        self.toPrimePermanences  = []         # The permanence connections to the toPrime segments.

        # FSegment portion--------
        self.terminalSynapses = []
        for tCol in range( columnDimensions ):
            if BinarySearch( terminalColumnsList, tCol ):
                newSynapseBundle = SynapseBundle( cellsPerColumn, initialPermanence )
            else:
                newSynapseBundle = SynapseBundle( cellsPerColumn, 0.0 )

            self.terminalSynapses.append( newSynapseBundle )

        self.incidentSynapses = []
        for iCol in range( columnDimensions ):
            if BinarySearch( incidentColumnsList, iCol ):
                newSynapseBundle = SynapseBundle( cellsPerColumn, initialPermanence )
            else:
                newSynapseBundle = SynapseBundle( cellsPerColumn, 0.0 )

            self.incidentSynapses.append( newSynapseBundle )

        self.activeBundlesTerminal = terminalColumnsList
        self.activeBundlesIncident = incidentColumnsList

        # Vector portion---------
        self.dimensions = len( vector )      # Number of dimensions vector positions will be in.
        self.vector     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            posI = ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 )
            self.vector.append( posI )

        self.self_record = []

#    def __repr__( self ):
#    # Returns string properties of the segment.
#
#        self.self_record.append( "-Program Completion-" )
#        self.self_record.append( self.AddStateToRecord() )
#
#        final_string = ""
#        for line in self.self_record:
#            final_string = final_string + line + "\n"
#
#        return final_string

    def AddStateToRecord( self ):
    # Adds details of current state, as a string, to record to be printed upon program completion.

        activeBundlesTerminalString = []
        for actBTer in self.activeBundlesTerminal:
            activeBundlesTerminalString.append( ( actBTer, self.terminalSynapses[ actBTer ].synapses ) )

        activeBundlesIncidentString = []
        for actBIns in self.activeBundlesIncident:
            activeBundlesIncidentString.append( ( actBIns, self.incidentSynapses[ actBIns ].synapses ) )

        primedSegString = []
        for index, pSeg in enumerate( self.toPrime ):
            primedSegString.append( ( pSeg, self.toPrimePermanences[ index ] ) )

        return ( "< Vector: %s, \n Active: %s, Last Active: %s, Time Since Active: %s, Activation Threshold: %s \n Active Incident Columns: %s, %s, \n Active Terminal Columns: %s, %s \n Primed Connections: %s > \n"
            % (self.vector, self.active, self.lastActive, self.timeSinceActive, self.activationThreshold, len( self.activeBundlesIncident ), activeBundlesIncidentString, len( self.activeBundlesTerminal ), activeBundlesTerminalString, primedSegString ) )

    def AddToPrime( self, segmentsToAdd, initialPermanence, permanenceIncrement, permanenceDecay ):
    # If segmentsToAdd isn't in self.toPrime then adds it with initialPermanence.
    # If segmentsToAdd is in then support its permanence.
    # Decay all permanences below 1.0.

        self.self_record.append( "ADD TO PRIME SEGMENT" )

        for addSeg in segmentsToAdd:
            idx = bisect_left( self.toPrime, addSeg )

            # Support if exists.
            if idx != len( self.toPrime ) and self.toPrime[ idx ] == addSeg:
                self.toPrimePermanences[ idx ] += permanenceIncrement
                self.self_record.append( "Support permanence: " + str( addSeg ) )

            # Create new if doesn't exist.
            else:
                self.toPrime.insert( idx, addSeg )
                self.toPrimePermanences.insert( idx, initialPermanence )
                self.self_record.append( "New permanence: " + str( addSeg ) )

        # Decay all.
        toDelete = []
        for index in range( len( self.toPrime ) ):
            if self.toPrimePermanences[ index ] >= 1.0:
                self.toPrimePermanences[ index ] = 1.0
            else:
                self.toPrimePermanences[ index ] -= permanenceDecay
                if self.toPrimePermanences[ index ] <= 0.0:
                    toDelete.insert( 0, index )
        for toDel in toDelete:
            del self.toPrime[ toDel ]
            del self.toPrimePermanences[ toDel ]
            self.self_record.append( "Deleted: " + str( toDel ) )

    def GetPrimed( self ):
    # Return the list of primed segments above threshold of 1.0.

        toReturn = []

        for index in range( len( self.toPrime ) ):
            if self.toPrimePermanences[ index ] >= 1.0:
                toReturn.append( self.toPrime[ index ] )

        self.self_record.append( "GET PRIMED SEGMENTS: " + str( toReturn ) )
        return toReturn

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

                for i in range( len( self.incidentSynapses ) ):
                    if ( self.incidentSynapses[ i ].getWinnerCell()[ 0 ] == other.incidentSynapses[ i ].getWinnerCell()[ 0 ]
                        and self.incidentSynapses[ i ].getWinnerCell()[ 1 ] > lowerThreshold
                        and other.incidentSynapses[ i ].getWinnerCell()[ 1 ] > lowerThreshold ):

                            incidentEquality += 1

                for ii in range( len( self.terminalSynapses ) ):
                    if ( self.terminalSynapses[ ii ].getWinnerCell()[ 0 ] == other.terminalSynapses[ ii ].getWinnerCell()[ 0 ]
                        and self.terminalSynapses[ ii ].getWinnerCell()[ 1 ] > lowerThreshold
                        and other.terminalSynapses[ ii ].getWinnerCell()[ 1 ] > lowerThreshold ):

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

    def CheckOverlap( self, FCellList, cellsPerColumn, synapsesToCheck, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        self.self_record.append( "CHECK OVERLAP" )

        overlap = 0
        passingCells = []
        failingCells = []

        for FCell in FCellList:
            col  = int( FCell / cellsPerColumn )
            cell = int( FCell % cellsPerColumn )

            if synapsesToCheck[ col ].active and synapsesToCheck[ col ].ReturnTrueIfActive( cell, lowerThreshold ):
                passingCells.append( ( col, cell ) )
                overlap += 1

            else:
                failingCells.append( ( col, cell ) )

        self.self_record.append( "FCell List: " + str( FCellList ) )
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

        if self.CheckOverlap( activeFCells, cellsPerColumn, self.incidentSynapses, lowerThreshold ) and self.Inside( vector ):
            return True
        else:
            return False

    def CheckIfPredicting( self, activeFCells, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this segments is still predicted by checking terminal overlap and vector.

        if self.CheckOverlap( activeFCells, cellsPerColumn, self.terminalSynapses, lowerThreshold ) and self.Inside( vector ):
            return True
        else:
            return False

    def ReturnTerminalFCells( self, cellsPerColumn, lowerThreshold ):
    # Return the active terminal cells for this segment in a list format.

        terminalFCellList = []

        for synapseBundleIndex in range( len( self.terminalSynapses ) ):
            if self.terminalSynapses[ synapseBundleIndex ].active:

                cellList = self.terminalSynapses[ synapseBundleIndex ].ReturnActiveCells( lowerThreshold )
                for cellIndex in cellList:

                    terminalFCellList.append( cellIndex + ( synapseBundleIndex * cellsPerColumn ) )

        return terminalFCellList

    def AdjustThreshold( self, minActivation, maxActivation ):
    # Check the number of bundles that have selected winner. Use this to adjust activationThreshold.

        numWinners = 0

        for terB in self.terminalSynapses:
            if terB.chosenWinner:
                numWinners += 1
        for incB in self.incidentSynapses:
            if incB.chosenWinner:
                numWinners += 1

        self.activationThreshold = ( ( minActivation - maxActivation ) * np.exp( -1 * numWinners ) ) + maxActivation
        self.self_record.append( "ADJUST THRESHOLD: " + str( self.activationThreshold ) )

    def DecayAndCreateBundles( self, lastActiveCols, activeCols, permanenceDecay, initialPermanence, maxBundlesToAddPer ):
    # If segment doesn't have an active terminal bundle to an active column then remove an active bundle to a
    # non-active column and add this one.
    # If segment doesn't have an active incident bundle to an last-active column then remove an active bundle to a
    # non-active column and add this one.
    # Decay any incident synapses to non-active lastActiveCols, and terminal synapses on-active activeCols.

        self.self_record.append( "DECAY AND CREATE BUNDLES" )

        # Find the active and lastActive columns missing terminal and incident active bundles.
        toAddIncident = []
        for lCol in lastActiveCols:
            if not self.incidentSynapses[ lCol ].active:
                toAddIncident.append( lCol )
        toAddTerminal = []
        for pCol in activeCols:
            if not self.terminalSynapses[ pCol ].active:
                toAddTerminal.append( pCol )
        self.self_record.append( "To add Incident: " + str( toAddIncident ) )
        self.self_record.append( "To add Terminal: " + str( toAddTerminal ) )

        # Randomly select maxBundlesToAddPer of these missing ones.
        if len( toAddIncident ) > maxBundlesToAddPer:
            newIncident = sample( toAddIncident, maxBundlesToAddPer )
        else:
            newIncident = toAddIncident
        if len( toAddTerminal ) > maxBundlesToAddPer:
            newTerminal = sample( toAddTerminal, maxBundlesToAddPer )
        else:
            newTerminal = toAddTerminal

        # Create new active synpase bundles for these randomly selected ones.
        for newICol in newIncident:
            self.incidentSynapses[ newICol ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesIncident, newICol )
            self.self_record.append( "Add Incident Bundle at Column: " + str( newICol ) )
        for newTCol in newTerminal:
            self.terminalSynapses[ newTCol ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesTerminal, newTCol )
            self.self_record.append( "Add Terminal Bundle at Column: " + str( newTCol ) )

        # Decay all active synapse bundles by small amount.
        # If all synapses in bundle decay to 0.0 then make bundle inactive.
        toRemoveIncident = []
        toRemoveTerminal = []
        for actInc in self.activeBundlesIncident:
            if not self.incidentSynapses[ actInc ].SynapseDecay( permanenceDecay ):
                toRemoveIncident.append( actInc )
        for actTer in self.activeBundlesTerminal:
            if not self.terminalSynapses[ actTer ].SynapseDecay( permanenceDecay ):
                toRemoveTerminal.append( actTer )

        if len( toRemoveIncident ) > 0:
            for removeInc in toRemoveIncident:
                self.activeBundlesIncident.remove( removeInc )
                self.self_record.append( "Remove Incident Bundle at Column: " + str( removeInc ) )
        if len( toRemoveTerminal ) > 0:
            for removeTer in toRemoveTerminal:
                self.activeBundlesTerminal.remove( removeTer )
                self.self_record.append( "Remove Terminal Bundle at Column: " + str( removeTer ) )

    def FindTerminalWinner( self, column ):
    # Check the terminal synapse bundle at given column for the winner and return it plus the permenence
    # value as a tuple.

        winnerCell = self.terminalSynapses[ column ].getWinnerCell()

        self.self_record.append( "FIND TERMINAL WINNER" )
        self.self_record.append( "Column: " + str( column ) + " winnerCell: " + str( winnerCell ) )
        self.self_record.append( str( ( column, self.terminalSynapses[ column ].synapses ) ) )

        if not self.terminalSynapses[ column ].active:
            self.self_record.append( "Terminal bundle isn't active." )

        return winnerCell

    def FindIncidentWinner( self, column ):
    # Check the terminal synapse bundle at given column for the winner and return it plus the permenence
    # value as a tuple.

        winnerCell = self.incidentSynapses[ column ].getWinnerCell()

        self.self_record.append( "FIND INCIDENT WINNER" )
        self.self_record.append( "Column: " + str( column ) + " winnerCell: " + str( winnerCell ) )
        self.self_record.append( str( ( column, self.incidentSynapses[ column ].synapses ) ) )

        if not self.incidentSynapses[ column ].active:
            self.self_record.append( "Incident bundle isn't active." )

        return winnerCell

    def SupportTerminalWinner( self, column, winnerCell, permanenceIncrement, permanenceDecrement ):
    # For the given synapse bundle, support the winnerCell by permanenceIncrement,
    #  and decay the losers by permanenceDecrement.

        self.self_record.append( "SUPPORT TERMINAL WINNER" )
        self.self_record.append( "Column: " + str( column ) + " winnerCell: " + str( winnerCell ) )
        self.self_record.append( str( ( column, self.terminalSynapses[ column ].synapses ) ) )

        if self.terminalSynapses[ column ].active:
            self.terminalSynapses[ column ].SupportWinner( winnerCell, permanenceIncrement, permanenceDecrement )

        else:
            self.self_record.append( "Synapse bundle at this location doesn't have an active synapse." )

    def SupportIncidentWinner( self, column, winnerCell, permanenceIncrement, permanenceDecrement ):
    # For the given synapse bundle, support the winnerCell by permanenceIncrement,
    #  and decay the losers by permanenceDecrement.

        self.self_record.append( "SUPPORT INCIDENT WINNER" )
        self.self_record.append( "Column: " + str( column ) + " winnerCell: " + str( winnerCell ) )
        self.self_record.append( str( ( column, self.incidentSynapses[ column ].synapses ) ) )

        if self.incidentSynapses[ column ].active:
            self.incidentSynapses[ column ].SupportWinner( winnerCell, permanenceIncrement, permanenceDecrement )

        else:
            self.self_record.append( "Synapse bundle at this location doesn't have an active synapse." )

class FCell:

    def __init__( self, initialPermanence, numOCells, pctAllowedOCellConns ):
    # Create a new feature level cell with synapses to OCells.

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
