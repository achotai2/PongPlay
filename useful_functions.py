from bisect import bisect_left
from random import randrange

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
# Checks if vector1 is within checkRange of vector2.

    if len( vector1 ) != len( vector2 ):
        print( "Error: Vectors not of same length." )
        exit()

    for i in range( len( vector1 ) ):
        if vector1[ i ] > vector2[ i ] + checkRange or vector1[ i ] < vector2[ i ] - checkRange:
            return False

    return True

def FastIntersect( list1, list2 ):
# Computes the intersection of two sorted lists and returns it.

    i = 0
    j = 0
    intersection = []

    while i < len( list1 ) and j < len( list2 ):
        if list1[ i ] == list2[ j ]:
            intersection.append( list1[ i ] )
            i += 1
            j += 1

        elif list1[ i ] < list2[ j ]:
            i += 1

        elif list1[ i ] > list2[ j ]:
            j += 1

    return intersection

def ModThisSynapse( currentValue, howMuch, notIfOne ):
# Modify the synapse value and do checks, then return new value.

    newValue = currentValue

    if ( notIfOne and newValue == 1.0 ) or ( newValue == 0.0 and howMuch < 0.0 ):
        return newValue

    newValue += howMuch

    if newValue > 1.0:
        newValue = 1.0
    elif newValue < 0.0:
        newValue = 0.0

    return newValue

def GenerateUnitySDR( SDRList, maxReturnSDRSize, cellsPerColumn ):
# Given a list of SDRs generate a new list of requested size by choosing the most frequent cells, and random otherwise.
# This function assumes the SDRs are sorted lists, and each cell can only appear once.

    returnSDR = []
    cellsList = []
    countList = []

    # Go through SDRs and count occurrances of each cell.
    for SDR in SDRList:
        for cell in SDR:
            # First check if cell is in list already.
            i = bisect_left( cellsList, cell )

            # If it is then add to its count.
            if i != len( cellsList ) and cellsList[ i ] == cell:
                countList[ i ] += 1

            # If not then add it.
            else:
                cellsList.insert( i, cell )
                countList.insert( i, 1 )

    # Once we have the counts, arrange the cells by their count.
    sortedCells = []
    for x in range( len( SDRList ) ):
        sortedCells.append( [] )
    for index, cell in enumerate( cellsList ):
        sortedCells[ countList[ index ] - 1 ].append( cell )

    while len( returnSDR ) < maxReturnSDRSize:
        chosenCell = None
        for c in reversed( range( len( SDRList ) ) ):
            if len( sortedCells[ c ] ) > 0:
                chosenCell = sortedCells[ c ].pop( randrange( 0, len( sortedCells[ c ] ) ) )
                break
        if chosenCell == None:
            print( "Something went wrong in GenerateUnitySDR()" )
            exit()

        i = bisect_left( returnSDR, chosenCell )
        # Don't add the cell if we already have one from the same column.
        if i == len( returnSDR ):
            if i == 0 or int( returnSDR[ i - 1 ] / cellsPerColumn ) != int( chosenCell / cellsPerColumn ):
                returnSDR.append( chosenCell )
        elif int( returnSDR[ i ] / cellsPerColumn ) != int( chosenCell / cellsPerColumn ):
            if i == 0 or int( returnSDR[ i - 1 ] / cellsPerColumn ) != int( chosenCell / cellsPerColumn ):
                returnSDR.insert( i, chosenCell )

    return returnSDR