function image = ReadImage(group, X, k)
Img = zeros(28, 560);
pos = find(group == k);
ind = randperm(length(pos), 20);
L = length(ind);

for l = 1:L    
    Img(:, (28*(l-1) + 1):(28*l)) = reshape(X(pos(ind(l)), :), 28, 28);
end

image = imshow(Img);

end