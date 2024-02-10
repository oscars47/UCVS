#!/usr/bin/julia

function distance(mags1, times1, mags2, times2)
    times1 = times1 .- first(times1)
    times2 = times2 .- first(times2)

    # end_time = minimum([last(times1), last(times2)])
    # lc_1 = Lightcurve.get_subseq(lc_1, 0, end_time)
    # lc_2 = Lightcurve.get_subseq(lc_2, 0, end_time)

    if (length(mags1) != length(times1) || length(mags2) != length(times2))
        return 0
    end

    # check for empty lightcurves
    if (length(mags1) == 0 or length(mags2) == 0)
        return None
    end
    
    if (mags1 == mags2 && times1 == times2)
        return 0.0
    end

    # pushfirst!(mags1, 0)
    # pushfirst!(times1, 0)
    # pushfirst!(mags2, 0)
    # pushfirst!(times2, 0)

    n = length(mags1)
    m = length(mags2)

    DP = zeros(n, m)
    DP[1,:] .= 
    

end

A = [1.9, 2.7, 23.5]