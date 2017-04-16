d = 8192;
createnote = @(frq,dur) sin(2*pi* [1:dur]/d * (440*2.^((frq-1)/12)));
notes = {'A' 'A#' 'B' 'C' 'C#' 'D' 'D#' 'E' 'F' 'F#' 'G' 'G#'};
song = {'A' 'A' 'E' 'E' 'F#' 'F#' 'E' 'E' 'D' 'D' 'C#' 'C#' 'B' 'B' 'A' 'A'};
for k = 1:length(song)
    idx = strcmp(song(k), notes);
    songidx(k) = find(idx);
end    
dur = 0.35*d;
songnotes = [];
for k = 1:length(songidx)
    songnotes = [songnotes; [createnote(songidx(k),dur)]'];
end
soundsc(songnotes, d);
