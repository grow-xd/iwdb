import React from 'react';

export const metadata = {
    title: "About",
    description: "Here the users will find the about section of our project"
}

export default function AboutLayout({children}) {
    return (
        <div className="">
            {children}
        </div>
    )
}
