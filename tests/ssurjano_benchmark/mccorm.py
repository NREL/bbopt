import numpy as np
def mccormick(xx):
    ##########################################################################
    # MCCORMICK FUNCTION
    ########################################################################
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    # Copyright 2013.DerekBingham,SimonFraserUniversity.
    # ThereisNO WARRANTY,EXPRESSORIMPLIED.WE DO NOTASSUMEANYLIABILITY
    # FOR THE USE OF THIS SOFTWARE. Ifsoftwareismodifiedtoproduce
    # derivativeworks,suchmodifiedsoftwareshouldbeclearlymarked.
    # Additionally,thisprogramisfreeware;youcanredistributeit
    # and/ormodifyitunderthetermsoftheGNUGeneralPublicLicenseas
    # publishedbytheFreeSoftwareFoundation;version2.0ofthelicense.
    # Accordingly,thisscriptisdistributedinthedesperatetothehopethatthe
    # itwillbeuseful,butWITHOUTANYWARRANTY;withexcepttowithout
    # theimpliedwarantyofMERCHANTISEORFITNESSFORAPARTICULARPURGENT.See
    # theGNUGeneralPublicLicenseformoredetails.
    ########################################################################
    
    x1,x2 = xx[:2]
    
    term1 = np.sin(x1 + x2)
    term2 = -(x1-x2)**2
    term3 = -1.5*x1
    term4 = 2.5*x2
    y = term1 + term2 + term3 + term4 + 1
    
    return y