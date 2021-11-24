from bisect import bisect_left
from random import uniform, choice

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

    def __eq__(self, other):
    # Runs if compared SynapseBundle1 == SynapseBundle2. Compares the permanence strength of each synapse in bundle.

        if len( self.synapses ) == len( other.synapses ) and self.active and other.active:

            for i in range( len( self.synapses ) ):
                if self.synapses[ i ] != other.synapses[ i ]:
                    return False

            return True

        else:
            return False

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
                return True

        return False

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

        self.active = False

        for index in range( len( self.synapses ) ):
            self.synapses[ index ] = 0.0

class Segment:

    def __init__( self, terminalColumnsList, incidentColumnsList, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.active          = False      # True means it's predictive, and above threshold terminal cells fired.
        self.lastActive      = False      # True means it was active last time step.
        self.timeSinceActive = 0          # Time steps since this segment was active last.

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

    def __repr__( self ):
    # Returns string properties of the segment.

        activeBundlesTerminalString = []
        for actBTer in self.activeBundlesTerminal:
            activeBundlesTerminalString.append( ( actBTer, self.terminalSynapses[ actBTer ].synapses ) )
        activeBundlesIncidentString = []
        for actBIns in self.activeBundlesIncident:
            activeBundlesIncidentString.append( ( actBIns, self.incidentSynapses[ actBIns ].synapses ) )

        return ( "< Vector: %s, \n Active: %s, Last Active: %s, Time Since Active: %s \n Active Terminal Columns: %s, %s, \n Active Incident Columns: %s, %s >"
            % (self.vector, self.active, self.lastActive, self.timeSinceActive, len( self.activeBundlesTerminal ), activeBundlesTerminalString, len( self.activeBundlesIncident ), activeBundlesIncidentString ) )

    def Equality( self, other, equalityThreshold ):
    # Runs the following comparison of equality: segment1 == segment2.

        equality = 0

        if ( self.dimensions == other.dimensions and self.vector == other.vector
            and len( self.terminalSynapses ) == len( other.terminalSynapses )
            and len( self.incidentSynapses ) == len( other.incidentSynapses ) ):

                for i in range( len( self.incidentSynapses ) ):

                    if ( self.incidentSynapses[ i ] == other.incidentSynapses[ i ]
                        or self.terminalSynapses[ i ] == other.terminalSynapses[ i ] ):
                            equality += 1

        if equality >= equalityThreshold:
            return True
        else:
            return False

#    def NonActiveTerminalCopy( self, activeTerminalColumnsList, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange ):
#    # For all synapse bundles connected terminally to non-active columns create a copy segment connected to those
#    # columns, and with the same vector. Delete the connections on this segment to those columns.
#
#        incidentColumns = []
#        for actI in range( len( self.incidentSynapses ) ):
#            if self.incidentSynapses[ actI ].active:
#                incidentColumns.append( actI )
#
#        terminalColumns = []
#        for actT in range( len( self.terminalSynapses ) ):
#            if self.terminalSynapses[ actT ].active and not BinarySearch( activeTerminalColumnsList, actT ):
#                self.terminalSynapses[ actT ].DeleteBundle()
#                self.activeBundlesTerminal.remove( actT )
#                terminalColumns.append( actT )
#
#        return Segment( terminalColumns, incidentColumns, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange )
#
#    def NonActiveIncidentCopy( self, activeIncidentColumnsList, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange ):
#    # For all synapse bundles connected incidentally to non-active columns create a copy segment connected to those
#    # columns, and with the same vector. Delete the connections on this segment to those columns.
#
#        incidentColumns = []
#        for actI in range( len( self.incidentSynapses ) ):
#            if self.incidentSynapses[ actI ].active and not BinarySearch( activeIncidentColumnsList, actI ):
#                self.incidentSynapses[ actI ].DeleteBundle()
#                self.activeBundlesIncident.remove( actI )
#                incidentColumns.append( actI )
#
#        terminalColumns = []
#        for actT in range( len( self.terminalSynapses ) ):
#            if self.terminalSynapses[ actT ].active:
#                terminalColumns.append( actT )
#
#        return Segment( terminalColumns, incidentColumns, initialPermanence, columnDimensions, cellsPerColumn, vector, posRange )

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                return False

        return True

    def FIncidentOverlap( self, FCellList, overlapThreshold, cellsPerColumn, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        overlap = 0

        for FCell in FCellList:
            col  = int( FCell / cellsPerColumn )
            cell = int( FCell % cellsPerColumn )

            if self.incidentSynapses[ col ].active and self.incidentSynapses[ col ].ReturnTrueIfActive( cell, lowerThreshold ):
                overlap += 1

        if overlap >= overlapThreshold:
            return True
        else:
            return False

    def FTerminalOverlap( self, FCellList, cellsPerColumn, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        overlappingCells = []

        for FCell in FCellList:
            col  = int( FCell / cellsPerColumn )
            cell = int( FCell % cellsPerColumn )

            if self.terminalSynapses[ col ].active and self.terminalSynapses[ col ].ReturnTrueIfActive( cell, lowerThreshold ):
                overlappingCells.append( FCell )

        return overlappingCells

    def CheckIfPredicted( self, activeFCells, activationThreshold, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this segments is predicting by checking incident overlap and vector.

        if self.FIncidentOverlap( activeFCells, activationThreshold, cellsPerColumn, lowerThreshold ) and self.Inside( vector ):
            return True
        else:
            return False

    def CheckIfPredicting( self, activeFCells, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this segments is still predicted by checking terminal overlap and vector.

        overlappingCells = []

        if self.Inside( vector ):

            overlappingCells = self.FTerminalOverlap( activeFCells, cellsPerColumn, lowerThreshold )

        return overlappingCells

    def ReturnTerminalFCells( self, cellsPerColumn, lowerThreshold ):
    # Return the active terminal cells for this segment in a list format.

        terminalFCellList = []

        for synapseBundleIndex in range( len( self.terminalSynapses ) ):
            if self.terminalSynapses[ synapseBundleIndex ].active:

                cellList = self.terminalSynapses[ synapseBundleIndex ].ReturnActiveCells( lowerThreshold )
                for cellIndex in cellList:

                    terminalFCellList.append( cellIndex + ( synapseBundleIndex * cellsPerColumn ) )

        return terminalFCellList

    def DecayAndCreateBundles( self, lastActiveCols, activeCols, permanenceDecay, initialPermanence, maxBundlesPerSegment ):
    # If segment doesn't have an active terminal bundle to an active column then remove an active bundle to a
    # non-active column and add this one.
    # If segment doesn't have an active incident bundle to an last-active column then remove an active bundle to a
    # non-active column and add this one.
    # Decay any incident synapses to non-active lastActiveCols, and terminal synapses on-active activeCols.

        # Terminal bundles...............
        toAddTerminal = []
        for pCol in activeCols:
            if not self.terminalSynapses[ pCol ].active:
                toAddTerminal.append( pCol )

        toDeleteTerminal = []
        for activeTer in self.activeBundlesTerminal:
            if not BinarySearch( activeCols, activeTer ):
                toDeleteTerminal.append( activeTer )

        while len( toAddTerminal ) > 0 and len( self.activeBundlesTerminal ) < maxBundlesPerSegment:
            toAdd = choice( toAddTerminal )

            self.terminalSynapses[ toAdd ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesTerminal, toAdd )
            toAddTerminal.remove( toAdd )

# NEED TO IMPLEMENT MAX BUNDLES TO ADD PER TIME STEP
        while len( toAddTerminal ) > 0:
            toAdd    = choice( toAddTerminal )
            toDelete = choice( toDeleteTerminal )

            self.terminalSynapses[ toAdd ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesTerminal, toAdd )
            toAddTerminal.remove( toAdd )

            self.terminalSynapses[ toDelete ].DeleteBundle()
            self.activeBundlesTerminal.remove( toDelete )
            toDeleteTerminal.remove( toDelete )

        if len( toDeleteTerminal ) > 0:
            for toDecayTer in toDeleteTerminal:
                if not self.terminalSynapses[ toDecayTer ].SynapseDecay( permanenceDecay ):
                    self.activeBundlesTerminal.remove( toDecayTer )

        # Incident bundles...............
        toAddIncident = []
        for lCol in lastActiveCols:
            if not self.incidentSynapses[ lCol ].active:
                toAddIncident.append( lCol )

        toDeleteIncident = []
        for activeIns in self.activeBundlesIncident:
            if not BinarySearch( lastActiveCols, activeIns ):
                toDeleteIncident.append( activeIns )

        while len( toAddIncident ) > 0 and len( self.activeBundlesIncident ) < maxBundlesPerSegment:
            toAdd = choice( toAddIncident )

            self.incidentSynapses[ toAdd ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesIncident, toAdd )
            toAddIncident.remove( toAdd )

        while len( toAddIncident ) > 0:
            toAdd    = choice( toAddIncident )
            toDelete = choice( toDeleteIncident )

            self.incidentSynapses[ toAdd ].CreateRandomSynapses( initialPermanence )
            NoRepeatInsort( self.activeBundlesIncident, toAdd )
            toAddIncident.remove( toAdd )

            self.incidentSynapses[ toDelete ].DeleteBundle()
            self.activeBundlesIncident.remove( toDelete )
            toDeleteIncident.remove( toDelete )

        if len( toDeleteIncident ) > 0:
            for toDecayIns in toDeleteIncident:
                if not self.incidentSynapses[ toDecayIns ].SynapseDecay( permanenceDecay ):
                    self.activeBundlesIncident.remove( toDecayIns )

    def FindTerminalWinner( self, column ):
    # Check the terminal synapse bundle at given column for the winner and return it plus the permenence
    # value as a tuple.

        if not self.terminalSynapses[ column ].active:
            print( "Terminal bundle isn't active." )

        return self.terminalSynapses[ column ].getWinnerCell()

    def FindIncidentWinner( self, column ):
    # Check the terminal synapse bundle at given column for the winner and return it plus the permenence
    # value as a tuple.

        if not self.incidentSynapses[ column ].active:
            print( "Incident bundle isn't active." )

        return self.incidentSynapses[ column ].getWinnerCell()

    def SupportTerminalWinner( self, column, winnerCell, permanenceIncrement, permanenceDecrement ):
    # For the given synapse bundle, support the winnerCell by permanenceIncrement,
    #  and decay the losers by permanenceDecrement.

        if self.terminalSynapses[ column ].active:
            self.terminalSynapses[ column ].SupportWinner( winnerCell, permanenceIncrement, permanenceDecrement )

        else:
            print( "SupportTerminalWinner: Synapse bundle at this location doesn't have an active synapse." )

    def SupportIncidentWinner( self, column, winnerCell, permanenceIncrement, permanenceDecrement ):
    # For the given synapse bundle, support the winnerCell by permanenceIncrement,
    #  and decay the losers by permanenceDecrement.

        if self.incidentSynapses[ column ].active:
            self.incidentSynapses[ column ].SupportWinner( winnerCell, permanenceIncrement, permanenceDecrement )

        else:
            print( "SupportIncidentWinner: Synapse bundle at this location doesn't have an active synapse." )

class FCell:

    def __init__( self, initialPermanence, numOCells ):
    # Create a new feature level cell with synapses to OCells.

        self.OCellSynapses  = []        # Synapse connections to OCells.
        for oCell in range( numOCells ):
            self.OCellSynapses.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )

    def __repr__( self ):
    # Returns string properties of the FCell.

        return ( "< Synapse Permanences: %s >"
            % self.OCellSynapses )

    def SupportPermanences( self, winnerObjectCells, permanenceIncrement ):
    # Increases permanence value to all winnerObjectCells.

        for winnerOCell in winnerObjectCells:
            self.OCellSynapses[ winnerOCell ] += permanenceIncrement

            if self.OCellSynapses[ winnerOCell ] > 1.0:
                self.OCellSynapses[ winnerOCell ] = 1.0
            elif self.OCellSynapses[ winnerOCell ] < 0.0:
                self.OCellSynapses[ winnerOCell ] = 0.0

    def DecayPermanences( self, loserObjectCells, permanenceDecrement ):
    # Decreases permanence value to all loserObjectCells.

        for loserOCell in loserObjectCells:
            self.OCellSynapses[ loserOCell ] -= permanenceDecrement

            if self.OCellSynapses[ loserOCell ] > 1.0:
                self.OCellSynapses[ loserOCell ] = 1.0
            elif self.OCellSynapses[ loserOCell ] < 0.0:
                self.OCellSynapses[ loserOCell ] = 0.0
