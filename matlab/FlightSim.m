% extract data from preprocessed csv
T = readtable("..\out\adsb\data\history\adsb_trajectory_data_icao24_A733DD_waypoints.csv");
timestamps = datetime(T.timestamp_utc, ...
    "InputFormat", "yyyy-MM-dd'T'HH:mm:ss'Z'", ...
    "TimeZone", "UTC");
timeOfArrival = T.time_of_arrival_s;
waypoints = [T.latitude_deg, T.longitude_deg, T.altitude_m];

% set simulation times
startTime = timestamps(1);
stopTime = timestamps(end);
sampleTime = 10;     

% create simulation
sc = satelliteScenario(startTime,stopTime,sampleTime);
viewer = satelliteScenarioViewer(sc);

trajectory = geoTrajectory(waypoints,timeOfArrival,AutoPitch=true,AutoBank=true);

% create aircraft platform
aircraft = platform(sc,trajectory, Name="Aircraft", Visual3DModel="NarrowBodyAirliner.glb");
camtarget(viewer,aircraft);