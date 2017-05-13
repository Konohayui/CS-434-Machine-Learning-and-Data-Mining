function image = ReadImage(group, X, k)
Img = [];
pos = find(group == k);

for l = 1:20
    Img = [Img reshape(X(pos(l), :), 28, 28)];
end

image = imshow(Img);

end